/**
 * A Socket.IO-like client that works over plain HTTP, for serverless hosting where websockets
 * are not available. It exposes the small part of the Socket.IO API the app uses
 * (on / off / once / emit / connect / disconnect / connected / id) and talks to the
 * backend's /events/poll and /events/emit routes.
 */

type Listener = (...args: any[]) => void;

interface PolledEvent {
  id: number;
  name: string;
  data: any;
  sender?: string | null;
}

// Poll quickly while the user is looking at the page, slowly otherwise. Every poll is one function call.
const ACTIVE_INTERVAL_MS = 2500;
const HIDDEN_INTERVAL_MS = 15000;
// Stay in "fast" mode for a while after something happened (an upload, a progress event).
const BUSY_WINDOW_MS = 30000;
const IDLE_INTERVAL_MS = 6000;

export class PollingSocket {
  readonly id: string;
  connected = false;
  // Same shape as socket.io-client's Socket.io for code that reads `socket.io.opts`.
  readonly io = { opts: {} as Record<string, unknown> };

  private readonly baseUrl: string;
  private readonly headers: Record<string, string>;
  private readonly listeners = new Map<string, Set<Listener>>();
  private lastId = -1;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private running = false;
  private lastActivity = 0;
  private failures = 0;

  constructor(baseUrl: string, headers: Record<string, string> = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.headers = headers;
    this.id = `poll-${Math.random().toString(36).slice(2, 10)}`;
    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', this.handleVisibility);
    }
  }

  on(name: string, listener: Listener): this {
    if (!this.listeners.has(name)) this.listeners.set(name, new Set());
    this.listeners.get(name)!.add(listener);
    return this;
  }

  once(name: string, listener: Listener): this {
    const wrapper: Listener = (...args) => {
      this.off(name, wrapper);
      listener(...args);
    };
    return this.on(name, wrapper);
  }

  off(name: string, listener?: Listener): this {
    if (!listener) this.listeners.delete(name);
    else this.listeners.get(name)?.delete(listener);
    return this;
  }

  removeAllListeners(): this {
    this.listeners.clear();
    return this;
  }

  emit(name: string, data?: unknown): this {
    // Local listeners hear a client's own events only through the server, matching Socket.IO relays.
    this.markActive();
    fetch(`${this.baseUrl}/events/emit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...this.headers },
      body: JSON.stringify({ name, data: data ?? {}, sender: this.id }),
    })
      .then(() => this.schedule(0))
      .catch(() => undefined);
    return this;
  }

  connect(): this {
    if (this.running) return this;
    this.running = true;
    this.schedule(0);
    return this;
  }

  disconnect(): this {
    this.running = false;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    if (this.connected) {
      this.connected = false;
      this.dispatch('disconnect', 'client disconnect');
    }
    return this;
  }

  private handleVisibility = () => {
    if (this.running && !document.hidden) this.schedule(0);
  };

  private markActive() {
    this.lastActivity = Date.now();
  }

  private nextDelay(): number {
    if (typeof document !== 'undefined' && document.hidden) return HIDDEN_INTERVAL_MS;
    const busy = Date.now() - this.lastActivity < BUSY_WINDOW_MS;
    const base = busy ? ACTIVE_INTERVAL_MS : IDLE_INTERVAL_MS;
    // Back off while the server is unreachable.
    return Math.min(base * 2 ** Math.min(this.failures, 4), 60000);
  }

  private schedule(delay: number) {
    if (!this.running) return;
    if (this.timer) clearTimeout(this.timer);
    this.timer = setTimeout(() => this.poll(), delay);
  }

  private dispatch(name: string, ...args: any[]) {
    this.listeners.get(name)?.forEach((listener) => {
      try {
        listener(...args);
      } catch (error) {
        console.error(`[PollingSocket] listener for "${name}" failed`, error);
      }
    });
  }

  private async poll() {
    if (!this.running) return;
    try {
      const response = await fetch(`${this.baseUrl}/events/poll?since=${this.lastId}`, {
        headers: this.headers,
        cache: 'no-store',
      });
      if (!response.ok) throw new Error(`poll failed: ${response.status}`);
      const body: { events: PolledEvent[]; last_id: number } = await response.json();

      this.failures = 0;
      if (!this.connected) {
        this.connected = true;
        this.dispatch('connect');
      }
      this.lastId = body.last_id;
      if (body.events.length > 0) this.markActive();
      for (const event of body.events) {
        // Events relayed from this very client are not echoed back (Socket.IO's include_self=False).
        if (event.sender && event.sender === this.id) continue;
        this.dispatch(event.name, event.data);
      }
    } catch {
      this.failures += 1;
      if (this.connected) {
        this.connected = false;
        this.dispatch('disconnect', 'transport error');
      }
      this.dispatch('connect_error', new Error('event polling failed'));
    } finally {
      this.schedule(this.nextDelay());
    }
  }
}
