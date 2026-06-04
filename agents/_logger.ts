/**
 * Shared logger factory for agent handlers.
 * Creates a logger with a consistent [prefix][timestamp] format.
 */
export function createLogger(prefix: string) {
  return {
    log(...args: unknown[]) {
      console.log(`[${prefix}][${new Date().toISOString()}]`, ...args);
    },
    error(...args: unknown[]) {
      console.error(`[${prefix}][${new Date().toISOString()}]`, ...args);
    },
  };
}
