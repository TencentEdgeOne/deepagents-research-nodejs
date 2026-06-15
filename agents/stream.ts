/**
 * Deep Research Agent — EdgeOne Makers handler.
 *
 * Architecture: Lead Researcher delegates sub-questions to Expert Researcher
 * subagents (with web_search), then synthesizes a final answer.
 *
 * Streaming: uses agentInstance.stream() with streamMode: ["updates", "messages"]
 * and subgraphs: true. Description-to-card matching at complete time uses
 * ToolMessage content prefix comparison.
 */

import { initChatModel, tool } from 'langchain';
import { modelRetryMiddleware, toolRetryMiddleware, toolCallLimitMiddleware } from 'langchain';
import { createDeepAgent, CompositeBackend, StateBackend, StoreBackend, type SubAgent } from 'deepagents';
import { AIMessageChunk, ToolMessage } from '@langchain/core/messages';

type Model = Awaited<ReturnType<typeof initChatModel>>;
type Agent = ReturnType<typeof createDeepAgent>;

interface Env {
  AI_GATEWAY_API_KEY: string;
  AI_GATEWAY_BASE_URL: string;
  AI_GATEWAY_MODEL: string;
}

import { createLogger } from './_logger';

const logger = createLogger('research-stream');

// ─── Singleton model & agent (lazy init) ───

let model: Model | null = null;
let agent: Agent | null = null;

function getEnv(contextEnv: Record<string, string | undefined> | undefined): Env {
  const source = contextEnv ?? {};
  const apiKey = source.AI_GATEWAY_API_KEY?.trim() || "";
  const baseUrl = source.AI_GATEWAY_BASE_URL?.trim() || "";
  if (!apiKey || !baseUrl) {
    throw new Error("Missing AI_GATEWAY_API_KEY or AI_GATEWAY_BASE_URL");
  }
  return {
    AI_GATEWAY_API_KEY: apiKey,
    AI_GATEWAY_BASE_URL: baseUrl,
    AI_GATEWAY_MODEL: source.AI_GATEWAY_MODEL?.trim() || "@makers/deepseek-v4-flash",
  };
}

async function getModel(env: Env): Promise<Model> {
  if (!model) {
    logger.log('Initializing model...');
    model = await initChatModel(env.AI_GATEWAY_MODEL, {
      modelProvider: 'openai',
      apiKey: env.AI_GATEWAY_API_KEY,
      configuration: {
        baseURL: env.AI_GATEWAY_BASE_URL,
      },
      temperature: 0,
      timeout: 300_000,
    });
  }
  return model;
}

function getAgent(modelInstance: Model, checkpointer: any, store: any, contextTools: any): Agent {
  if (!agent) {
    logger.log('Initializing research agent...');

    const today = new Date().toISOString().slice(0, 7);
    const webSearchTools = contextTools.toLangChainTools(tool, ['web_search']);

    const researcherSubagent: SubAgent = {
      name: 'researcher',
      description:
        'An expert researcher that answers a specific sub-question using web search.',
      systemPrompt:
        `You are an expert researcher. Today is ${today}.\n` +
        `CRITICAL: You MUST respond in the EXACT same language as your task description. If the task is in Chinese, your ENTIRE output must be in Chinese. If in English, respond in English.\n\n` +
        `Workflow:\n` +
        `1. Call web_search 3-5 times with different queries to gather information from multiple angles.\n` +
        `2. After your searches complete, IMMEDIATELY write your final summary. Do NOT call web_search again.\n\n` +
        `HARD LIMIT: You may call web_search AT MOST 5 times total. After finishing your searches, you MUST stop and write your summary — no exceptions, no "let me search more".\n\n` +
        `Output rules:\n` +
        `- After searching, output ONLY your summary text (under 600 Chinese characters or 400 English words).\n` +
        `- Do NOT narrate your search process (e.g. "Let me search...", "I will look for...").\n` +
        `- Do NOT echo raw JSON from tool results.\n` +
        `- Do NOT say you want to search more. Just write the summary.`,
      tools: webSearchTools,
      middleware: [
        modelRetryMiddleware({ maxRetries: 3 }),
        toolRetryMiddleware({ maxRetries: 1, tools: ['web_search'] }),
        toolCallLimitMiddleware({
          toolName: 'web_search',
          runLimit: 15,
        }),
      ],
    };

    agent = createDeepAgent({
      model: modelInstance,
      systemPrompt:
        `You are a lead researcher. Today is ${today}.\n` +
        `CRITICAL: You MUST use the EXACT same language as the user. If the user writes in Chinese, ALL your output (plan text AND task descriptions) MUST be in Chinese. If in English, use English.\n\n` +
        `Process:\n` +
        `1. On your FIRST response, you MUST call the task tool to delegate 2-3 sub-questions. You may optionally include a brief plan sentence before the tool calls, but tool calls are MANDATORY in the first response.\n` +
        `2. Wait for ALL sub-agent results, then synthesize a concise final answer (under 400 English words or 600 Chinese characters).\n\n` +
        `Rules:\n` +
        `- Your first response MUST contain task tool calls. Never respond with only text and no tool calls.\n` +
        `- ALL task tool calls MUST happen in ONE single model response — batch them together.\n` +
        `- Do NOT dispatch additional tasks after receiving sub-agent results.\n` +
        `- Task descriptions MUST be in the user's language.\n` +
        `- Only use sub-agent findings. Do not fabricate.`,
      subagents: [researcherSubagent],
      middleware: [
        modelRetryMiddleware({ maxRetries: 3 }),
      ],
      checkpointer,
      store,
      backend: new CompositeBackend(
        new StateBackend(),
        {
          '/memories/': new StoreBackend({
            namespace: ['agent', 'memories'],
          }),
        },
      ),
      memory: ['/memories/AGENTS.md'],
    });
  }
  return agent;
}

// ─── SSE event shape ───

interface StreamEvent {
  type: string;
  source: 'main' | 'subagent';
  content?: string;
  name?: string;
  tool_name?: string;
  tool_call_id?: string;
  subagent_type?: string;
  description?: string;
  subagent_id?: string;
  args?: string;
}

// ─── SSE event stream generator ───

async function* eventStream(
  agentInstance: Agent,
  message: string,
  conversationId: string,
  signal?: AbortSignal,
): AsyncGenerator<string> {
  const knownSubagents = new Map<string, { saId: string }>();
  const pendingDescriptions = new Map<string, string>();
  const emittedToolCallIds = new Set<string>();
  const emittedToolResultIds = new Set<string>();
  const toolCallIdToName = new Map<string, string>();

  function extractNsSegment(ns: string[]): string {
    return ns.find((s) => s.startsWith('tools:')) ?? '';
  }

  function shortId(nsSegment: string): string {
    return nsSegment.split(':').pop()?.slice(0, 8) ?? '';
  }

  function ensureSubagent(nsSegment: string): { saId: string } {
    if (knownSubagents.has(nsSegment)) return knownSubagents.get(nsSegment)!;
    const saId = shortId(nsSegment);
    knownSubagents.set(nsSegment, { saId });
    return { saId };
  }

  function send(event: StreamEvent): string {
    return `data: ${JSON.stringify(event)}\n\n`;
  }

  try {
    const stream = await agentInstance.stream(
      { messages: [{ role: 'user', content: message }] },
      {
        configurable: { thread_id: conversationId },
        streamMode: ['updates', 'messages'],
        subgraphs: true,
        signal,
      } as any,
    );

    for await (const tuple of stream) {
      if (signal?.aborted) break;

      const [chunkNs, chunkType, chunkData] = tuple as any as [string[], string, any];
      const nsSegment = extractNsSegment(chunkNs);
      const isSubagent = !!nsSegment;

      // ── "updates" mode: lifecycle + tool events ──
      if (chunkType === 'updates') {
        const data: Record<string, any> = chunkData ?? {};

        for (const [nodeName, nodeData] of Object.entries(data)) {
          // (A) Main agent model_request → task tool_calls
          if (!isSubagent && nodeName === 'model_request') {
            const messages = (nodeData as any)?.messages ?? [];
            for (const msg of messages) {
              for (const tc of msg?.tool_calls ?? []) {
                if (tc.name !== 'task') continue;
                const desc = (tc.args?.description ?? '').slice(0, 500);
                const saType = tc.args?.subagent_type ?? 'researcher';
                pendingDescriptions.set(tc.id, desc);
                yield send({
                  type: 'subagent_pending',
                  source: 'main',
                  tool_call_id: tc.id,
                  subagent_type: saType,
                  description: desc,
                });
              }
            }
          }

          // (B) Subagent namespace events
          if (isSubagent) {
            const { saId } = ensureSubagent(nsSegment);

            yield send({
              type: 'subagent_step',
              source: 'subagent',
              subagent_id: saId,
            });

            // (B1) Subagent model_request → tool_calls
            if (nodeName === 'model_request') {
              const stateMessages = (nodeData as any)?.messages ?? [];
              for (const msg of stateMessages) {
                for (const tc of (msg as any)?.tool_calls ?? []) {
                  if (!tc?.name) continue;
                  const tcRealId: string = tc.id ?? '';
                  if (tcRealId && emittedToolCallIds.has(tcRealId)) continue;
                  if (tcRealId) {
                    emittedToolCallIds.add(tcRealId);
                    toolCallIdToName.set(tcRealId, tc.name);
                  }

                  const argsStr = typeof tc.args === 'string'
                    ? tc.args
                    : tc.args != null ? JSON.stringify(tc.args) : '';

                  yield send({
                    type: 'tool_call',
                    source: 'subagent',
                    name: tc.name,
                    subagent_id: saId,
                    tool_call_id: tcRealId,
                    ...(argsStr && { args: argsStr }),
                  });
                }
              }
            }

            // (B2) Subagent tools node → tool completion
            if (nodeName === 'tools') {
              const stateMessages = (nodeData as any)?.messages ?? [];
              for (const msg of stateMessages) {
                if (!ToolMessage.isInstance(msg) && (msg as any)?.type !== 'tool') continue;
                const toolTcId: string = (msg as any).tool_call_id ?? '';
                const resolvedName = msg.name ?? toolCallIdToName.get(toolTcId) ?? '';
                if (resolvedName === 'task') continue;
                if (toolTcId && emittedToolResultIds.has(toolTcId)) continue;
                if (toolTcId) emittedToolResultIds.add(toolTcId);

                yield send({
                  type: 'tool',
                  source: 'subagent',
                  tool_name: resolvedName,
                  subagent_id: saId,
                  tool_call_id: toolTcId,
                });
              }
            }
          }

          // (C) Main agent tools node → task ToolMessage → subagent_complete
          if (!isSubagent && nodeName === 'tools') {
            const messages = (nodeData as any)?.messages ?? [];
            for (const msg of messages) {
              if (msg.type !== 'tool' || msg.name !== 'task') continue;
              const taskToolCallId = msg.tool_call_id ?? '';

              let contentText = '';
              if (typeof msg.content === 'string') {
                contentText = msg.content;
              } else if (Array.isArray(msg.content)) {
                contentText = msg.content
                  .filter((block: any) => block.type === 'text')
                  .map((block: any) => block.text || '')
                  .join('');
              }

              yield send({
                type: 'subagent_complete',
                source: 'main',
                tool_call_id: taskToolCallId,
                description: pendingDescriptions.get(taskToolCallId) || '',
                content: contentText.slice(0, 100),
              });
            }
          }
        }

        continue;
      }

      // ── "messages" mode: stream text tokens ──
      if (chunkType === 'messages') {
        const [msg] = chunkData;
        if (!AIMessageChunk.isInstance(msg)) continue;
        if (!msg.text || msg.tool_call_chunks?.length) continue;

        const content = msg.text.replace(/\n{3,}/g, '\n\n');
        if (!content) continue;

        if (isSubagent) {
          const { saId } = ensureSubagent(nsSegment);
          yield send({
            type: 'ai',
            source: 'subagent',
            content,
            subagent_id: saId,
          });
        } else {
          yield send({ type: 'ai', source: 'main', content });
        }
      }
    }

    // ── Post-stream: check for unstreamed errors in final state ──
    try {
      const finalState = await agentInstance.graph.getState({ configurable: { thread_id: conversationId } });
      const msgs = finalState?.values?.messages || [];
      const lastMsg = msgs.length > 0 ? msgs[msgs.length - 1] : null;
      if (lastMsg) {
        const msgType = typeof lastMsg._getType === 'function' ? lastMsg._getType() : lastMsg.type;
        if (msgType === 'ai') {
          let text = '';
          if (typeof lastMsg.content === 'string') text = lastMsg.content;
          else if (Array.isArray(lastMsg.content)) {
            text = lastMsg.content.filter((p: any) => p.type === 'text').map((p: any) => p.text || '').join('');
          }
          if (text && text.includes('MiddlewareError')) {
            logger.error(`MiddlewareError in final state: ${text.slice(0, 200)}`);
            yield send({ type: 'error', source: 'main', content: text });
          }
        }
      }
    } catch {}

  } catch (e: unknown) {
    const error = e as Error;
    if (error.name === 'AbortError' || signal?.aborted) {
      logger.log('Stream aborted by user');
    } else {
      logger.error('Stream error:', error.message);
      yield send({ type: 'error', source: 'main', content: `Stream error: ${error.constructor.name}: ${String(error.message).slice(0, 500)}` });
    }
  }

  yield 'data: [DONE]\n\n';
}

// ─── EdgeOne Makers handler ───

export async function onRequest(context: any) {
  const { request, env, conversation_id: conversationId, run_id: runId } = context;
  logger.log('conversationId:', conversationId, 'runId:', runId);

  const body = request?.body ?? {};
  const action = body.action || 'chat';
  const signal = request?.signal as AbortSignal | undefined;

  const checkpointer = context.store.langgraphCheckpointer;
  const store = context.store.langgraphStore;

  // ── Delete conversation ──
  if (action === 'delete') {
    const threadId = body.conversationId;
    if (!threadId) {
      return new Response(JSON.stringify({ error: 'Missing conversationId' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json; charset=UTF-8' },
      });
    }
    try { await checkpointer.deleteThread(threadId); } catch {}
    return new Response(JSON.stringify({ deleted: true }), {
      status: 200,
      headers: { 'Content-Type': 'application/json; charset=UTF-8' },
    });
  }

  let agentInstance: Agent;
  try {
    const envVars = getEnv(env);
    const modelInstance = await getModel(envVars);
    agentInstance = getAgent(modelInstance, checkpointer, store, context.tools);
  } catch (e) {
    const msg = (e as Error).message;
    logger.error(msg);
    return new Response(JSON.stringify({ error: msg }), {
      status: 500,
      headers: { 'Content-Type': 'application/json; charset=UTF-8' },
    });
  }

  // ── History ──
  if (action === 'history') {
    const threadId = body.conversationId;
    logger.log('history request for threadId:', threadId);
    if (!threadId) {
      return new Response(JSON.stringify({ items: [] }), {
        status: 200,
        headers: { 'Content-Type': 'application/json; charset=UTF-8' },
      });
    }
    try {
      const state = await agentInstance.graph.getState({ configurable: { thread_id: threadId } });
      const rawMessages = state?.values?.messages || [];

      type HistoryItem =
        | { type: 'user'; content: string }
        | { type: 'coordinator'; content: string }
        | { type: 'subagentTask'; id: string; description: string; subagentType: string; content: string };

      const items: HistoryItem[] = [];
      const pendingTasks = new Map<string, { description: string; subagentType: string }>();

      for (const m of rawMessages) {
        const msgType = typeof m._getType === 'function' ? m._getType() : m.type;

        if (msgType === 'human') {
          const content = typeof m.content === 'string' ? m.content : '';
          if (content) items.push({ type: 'user', content });
          continue;
        }

        if (msgType === 'ai') {
          let textContent = '';
          if (typeof m.content === 'string') {
            textContent = m.content;
          } else if (Array.isArray(m.content)) {
            textContent = m.content.filter((p: any) => p.type === 'text').map((p: any) => p.text || '').join('');
          }

          for (const tc of (m.tool_calls || [])) {
            if (tc.name === 'task' && tc.id) {
              pendingTasks.set(tc.id, {
                description: (tc.args?.description || '').slice(0, 500),
                subagentType: tc.args?.subagent_type || 'researcher',
              });
            }
          }

          if (textContent) items.push({ type: 'coordinator', content: textContent });
          continue;
        }

        if (msgType === 'tool') {
          const toolCallId = m.tool_call_id || '';
          if (m.name === 'task' && pendingTasks.has(toolCallId)) {
            const taskInfo = pendingTasks.get(toolCallId)!;
            let rawContent = '';
            if (typeof m.content === 'string') {
              rawContent = m.content;
            } else if (Array.isArray(m.content)) {
              rawContent = m.content.filter((block: any) => block.type === 'text').map((block: any) => block.text || '').join('\n');
            }
            items.push({
              type: 'subagentTask',
              id: toolCallId,
              description: taskInfo.description,
              subagentType: taskInfo.subagentType,
              content: rawContent,
            });
            pendingTasks.delete(toolCallId);
          }
          continue;
        }
      }

      logger.log('history: found', items.length, 'items');
      return new Response(JSON.stringify({ items }), {
        status: 200,
        headers: { 'Content-Type': 'application/json; charset=UTF-8' },
      });
    } catch (e) {
      logger.error('history error:', (e as Error).message);
      return new Response(JSON.stringify({ items: [] }), {
        status: 200,
        headers: { 'Content-Type': 'application/json; charset=UTF-8' },
      });
    }
  }

  // ── Chat (SSE streaming) ──
  const { message } = body;
  logger.log('user message:', message);
  if (!message) {
    return new Response('Missing chat message', { status: 400 });
  }

  const encoder = new TextEncoder();
  const readableStream = new ReadableStream({
    async start(controller) {
      try {
        for await (const chunk of eventStream(agentInstance, message, conversationId, signal)) {
          if (signal?.aborted) break;
          controller.enqueue(encoder.encode(chunk));
        }
      } catch (e) {
        const error = e as Error;
        if (error.name === 'AbortError' || signal?.aborted) return;
        controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: 'error', source: 'main', content: error.message })}\n\n`));
      } finally {
        controller.close();
      }
    },
    cancel() {
      logger.log('Client disconnected');
    },
  });

  return new Response(readableStream, {
    status: 200,
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'X-Accel-Buffering': 'no',
    },
  });
}
