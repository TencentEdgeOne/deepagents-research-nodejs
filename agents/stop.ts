// Abort a specific active agent run by conversationId.

import { createLogger } from './_logger';

const logger = createLogger('stop');

export async function onRequest(context: any) {
  const { request } = context;
  const conversationId = request?.body?.conversationId as string | undefined;
  logger.log('conversationId:', conversationId);

  if (!conversationId) {
    logger.error('Missing conversationId');
    return new Response('Missing conversationId', { status: 400 });
  }

  const ret = context.utils.abortActiveRun(conversationId);
  logger.log('abortActiveRun result:', ret);

  const data = {
    status: ret?.aborted ? 'aborting' : 'idle',
    conversationId,
    ...ret,
  };

  return new Response(JSON.stringify(data), {
    status: 200,
    headers: { 'Content-Type': 'application/json; charset=UTF-8' },
  });
}
