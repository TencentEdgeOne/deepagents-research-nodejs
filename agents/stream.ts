/**
 * Deep Research Agent — EdgeOne Pages handler.
 *
 * Architecture: Lead Researcher delegates sub-questions to Expert Researcher
 * subagents (with web_search), then synthesizes a final answer.
 *
 * Streaming: iterates raw ProtocolEvents from streamEvents v3, mapping them to
 * SSE events for the frontend (subagent_pending, tool_call, ai, etc.).
 */

import { initChatModel, tool } from 'langchain';
import { modelRetryMiddleware, toolRetryMiddleware, toolCallLimitMiddleware } from 'langchain';
import { createDeepAgent, CompositeBackend, StateBackend, StoreBackend, type SubAgent } from 'deepagents';

type Model = Awaited<ReturnType<typeof initChatModel>>;
type Agent = ReturnType<typeof createDeepAgent>;

interface Env {
  AI_GATEWAY_API_KEY: string;
  AI_GATEWAY_BASE_URL: string;
}

import { createLogger } from './_logger';

const logger = createLogger('research-stream');

// ─── Singleton model & agent (lazy init) ───

let model: Model | null = null;
let agent: Agent | null = null;

function getEnv(contextEnv: Record<string, string | undefined> | undefined): Env {
  const source = contextEnv ?? {};
  const required = ['AI_GATEWAY_API_KEY', 'AI_GATEWAY_BASE_URL'] as const;
  const missing = required.filter((k) => !source[k]?.trim());

  if (missing.length) {
    throw new Error(`Missing environment variables: ${missing.join(', ')}`);
  }

  return {
    AI_GATEWAY_API_KEY: source.AI_GATEWAY_API_KEY!,
    AI_GATEWAY_BASE_URL: source.AI_GATEWAY_BASE_URL!,
  };
}

async function getModel(env: Env): Promise<Model> {
  if (!model) {
    logger.log('Initializing model...');
    model = await initChatModel('@makers/hy3-preview', {
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
        toolRetryMiddleware({ maxRetries: 2, tools: ['web_search'] }),
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
  function send(event: StreamEvent): string {
    return `data: ${JSON.stringify(event)}\n\n`;
  }

  // Guard against unhandled rejections from SDK internals (SubagentTransformer
  // rejects promises on tool-error that we don't consume via projections).
  const rejectionHandler = (reason: unknown) => {
    logger.log(`Swallowed unhandled rejection: ${String(reason).slice(0, 100)}`);
  };
  process.on('unhandledRejection', rejectionHandler);

  try {
    const run = await (agentInstance as any).streamEvents(
      { messages: [{ role: 'user', content: message }] },
      {
        version: 'v3',
        configurable: { thread_id: conversationId },
        signal,
      },
    );

    // Catch projection promise rejections we can reach directly
    const noop = () => {};
    if (run.output?.catch) run.output.catch(noop);
    if (run.subagents?.[Symbol.asyncIterator]) {
      (async () => {
        try {
          for await (const sa of run.subagents) {
            sa.output?.catch?.(noop);
          }
        } catch {}
      })();
    }

    // ── State ──
    const nsSegmentToCard = new Map<string, { cardId: string; saId: string }>();
    const taskToolCallIdToCard = new Map<string, { cardId: string; saId: string }>();
    const emittedToolCallIds = new Set<string>();
    const subagentTextBuffers = new Map<string, string>();
    let subagentCounter = 0;

    // ── Event loop ──
    for await (const event of run) {
      if (signal?.aborted) break;

      const ns: string[] = event.params?.namespace ?? [];
      const method: string = event.method ?? '';
      const data: any = event.params?.data ?? {};
      const depth = ns.length;
      const subagentNsSegment = (depth >= 1 && ns[0].startsWith('tools:')) ? ns[0] : '';

      // ── MESSAGES: text tokens ──
      if (method === 'messages') {
        const eventType: string = data.event ?? '';

        if (eventType === 'content-block-delta') {
          const delta = data.delta;
          if (!delta) continue;

          // Only accept text-delta; skip block-delta (tool_call streaming) etc.
          let text = '';
          if (typeof delta === 'object' && delta.type === 'text-delta') {
            text = delta.text ?? '';
          } else if (typeof delta === 'string') {
            text = delta;
          }
          if (!text) continue;

          const content = text.replace(/\n{3,}/g, '\n\n');
          if (!content) continue;

          // Subagent text (depth >= 2, known card)
          if (depth >= 2 && subagentNsSegment && nsSegmentToCard.has(subagentNsSegment)) {
            const card = nsSegmentToCard.get(subagentNsSegment)!;
            // Buffer subagent text per-card to strip JSON tool result echoes
            const bufKey = card.saId;
            if (!subagentTextBuffers.has(bufKey)) subagentTextBuffers.set(bufKey, '');
            subagentTextBuffers.set(bufKey, subagentTextBuffers.get(bufKey)! + content);

            // Flush buffer: strip JSON tool result echoes (individual objects and arrays)
            const buf = subagentTextBuffers.get(bufKey)!;
            const cleaned = buf
              .replace(/\[?\{["\u201c]title["\u201d]:.*?"engine":\s*"[^"]*"\s*\}[\],]*/g, '')
              .replace(/^\[[\s,]*\]/, '')
              .replace(/[\[\]]/g, '');
            // Only flush if we have enough to be confident JSON won't span next token
            // or if content ends with a sentence-ending char
            if (cleaned.length > 50 || /[。\.\n]$/.test(cleaned)) {
              subagentTextBuffers.set(bufKey, '');
              const trimmed = cleaned.replace(/\n{3,}/g, '\n\n').replace(/^[\s,\[\]]+/, '').replace(/[\s,\[\]]+$/, '');
              if (trimmed) {
                yield send({ type: 'ai', source: 'subagent', content: trimmed, subagent_id: card.saId, tool_call_id: card.cardId });
              }
            }

          } else if (depth <= 1 && (!subagentNsSegment || !nsSegmentToCard.has(subagentNsSegment))) {
            // Main agent text
            yield send({ type: 'ai', source: 'main', content });
          }
        }

        // Reset block tracking on new content-block-start (no-op now, kept for future use)
        if (eventType === 'content-block-start' && depth >= 2 && subagentNsSegment) {
          // placeholder
        }

        continue;
      }

      // ── TOOLS: lifecycle events ──
      if (method === 'tools') {
        const toolEvent: string = data.event ?? '';
        const toolCallId: string = data.tool_call_id ?? '';
        const toolName: string = data.tool_name ?? '';

        // Task started (depth <= 1) → register subagent
        if (depth <= 1 && toolName === 'task' && toolEvent === 'tool-started') {
          const rawInput = data.input;
          const input = typeof rawInput === 'string' ? JSON.parse(rawInput) : rawInput ?? {};
          const subagentType: string = input.subagent_type ?? input.agentName ?? 'researcher';
          const description: string = input.description ?? '';

          const saId = `sa-${++subagentCounter}`;
          const cardId = saId;
          const actualNsSegment = ns[0] ?? '';

          if (actualNsSegment) nsSegmentToCard.set(actualNsSegment, { cardId, saId });
          taskToolCallIdToCard.set(toolCallId, { cardId, saId });

          logger.log(`task-started: saId=${saId}, desc=${description.slice(0, 60)}`);

          yield send({ type: 'subagent_pending', source: 'main', tool_call_id: cardId, subagent_id: saId, subagent_type: subagentType, description: description.slice(0, 500) });
          yield send({ type: 'subagent_step', source: 'subagent', subagent_id: saId, tool_call_id: cardId });
          continue;
        }

        // Task finished (depth <= 1) → subagent complete
        if (depth <= 1 && toolEvent === 'tool-finished') {
          const card = taskToolCallIdToCard.get(toolCallId);
          if (card) {
            // Flush any remaining buffered text for this subagent
            const remaining = subagentTextBuffers.get(card.saId) || '';
            if (remaining) {
              const cleaned = remaining
                .replace(/\[?\{["\u201c]title["\u201d]:.*?"engine":\s*"[^"]*"\s*\}[\],]*/g, '')
                .replace(/^\[[\s,]*\]/, '')
                .replace(/[\[\]]/g, '')
                .replace(/\n{3,}/g, '\n\n')
                .replace(/^[\s,]+/, '');
              if (cleaned.trim()) {
                yield send({ type: 'ai', source: 'subagent', content: cleaned, subagent_id: card.saId, tool_call_id: card.cardId });
              }
              subagentTextBuffers.delete(card.saId);
            }
            logger.log(`task-finished: saId=${card.saId}`);
            yield send({ type: 'subagent_complete', source: 'main', tool_call_id: card.cardId, subagent_id: card.saId });
          }
          continue;
        }

        // Task error (depth <= 1) → subagent complete
        if (depth <= 1 && toolEvent === 'tool-error') {
          const card = taskToolCallIdToCard.get(toolCallId);
          if (card) {
            logger.error(`task-error: saId=${card.saId}, msg=${data.message}`);
            yield send({ type: 'subagent_complete', source: 'main', tool_call_id: card.cardId, subagent_id: card.saId });
          }
          continue;
        }

        // Subagent internal tools (depth >= 2)
        if (depth >= 2 && subagentNsSegment && toolName !== 'task') {
          const card = nsSegmentToCard.get(subagentNsSegment);
          if (!card) continue;

          if (toolEvent === 'tool-started') {
            if (toolCallId && emittedToolCallIds.has(toolCallId)) continue;
            if (toolCallId) emittedToolCallIds.add(toolCallId);

            const rawInput = data.input;
            const argsStr = rawInput != null
              ? (typeof rawInput === 'string' ? rawInput : JSON.stringify(rawInput))
              : '';

            yield send({ type: 'tool_call', source: 'subagent', name: toolName || '(unknown)', subagent_id: card.saId, tool_call_id: toolCallId, ...(argsStr && { args: argsStr }) });
          } else if (toolEvent === 'tool-finished' || toolEvent === 'tool-error') {
            yield send({ type: 'tool', source: 'subagent', tool_name: toolName || '(unknown)', subagent_id: card.saId, tool_call_id: toolCallId });
          }
        }
        continue;
      }

      // ── LIFECYCLE: subagent step events ──
      if (method === 'lifecycle' && depth >= 2 && subagentNsSegment) {
        const card = nsSegmentToCard.get(subagentNsSegment);
        if (card && data.event === 'started') {
          yield send({ type: 'subagent_step', source: 'subagent', subagent_id: card.saId, tool_call_id: card.cardId });
        }
      }
    }

    // Check for errors stored in final state (e.g. LLM quota exceeded)
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
      yield send({ type: 'error', source: 'main', content: `Stream error: ${error.constructor.name}: ${String(error.message).slice(0, 200)}` });
    }
  } finally {
    process.removeListener('unhandledRejection', rejectionHandler);
  }

  yield 'data: [DONE]\n\n';
}

// ─── EdgeOne Pages handler ───

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
