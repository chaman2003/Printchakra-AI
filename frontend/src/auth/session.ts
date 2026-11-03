/**
 * Login session kept in localStorage, plus helpers that attach it to API requests.
 *
 * The token goes out as `Authorization: Bearer ...`. Plain <img> tags and links cannot send headers,
 * so `withToken()` adds it as a query parameter for those.
 */

const TOKEN_KEY = 'printchakra_token';
const USER_KEY = 'printchakra_user';
export const AUTH_EXPIRED_EVENT = 'printchakra:auth-expired';

// Set once at startup (index.tsx). Kept out of an import so config.ts can use this module without a cycle.
let apiBase = '';
export const setApiBase = (base: string) => {
  apiBase = base;
};

export interface SessionUser {
  id: number;
  email: string;
  name: string;
  username?: string | null;
  role?: 'user' | 'admin';
}

// Per-user data kept in the browser. It must not survive a logout, or the next person on this device would see it.
const USER_DATA_KEYS = ['printchakra_ocr_results', 'printchakra_orchestrate_defaults'];

export const getToken = (): string => {
  try {
    return localStorage.getItem(TOKEN_KEY) || '';
  } catch {
    return '';
  }
};

export const getStoredUser = (): SessionUser | null => {
  try {
    const raw = localStorage.getItem(USER_KEY);
    return raw ? (JSON.parse(raw) as SessionUser) : null;
  } catch {
    return null;
  }
};

export const saveSession = (token: string, user: SessionUser) => {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, JSON.stringify(user));
};

export const clearSession = () => {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
  USER_DATA_KEYS.forEach((key) => localStorage.removeItem(key));
  sessionStorage.clear();
};

export const authHeader = (): Record<string, string> => {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
};

/** Adds the access token to a URL so it can be used in <img src> or window.open. */
export const withToken = (url: string): string => {
  const token = getToken();
  if (!token || !url.startsWith(apiBase)) return url;
  return `${url}${url.includes('?') ? '&' : '?'}access_token=${encodeURIComponent(token)}`;
};

const notifyExpired = () => window.dispatchEvent(new Event(AUTH_EXPIRED_EVENT));

/**
 * Wraps window.fetch once so every request to the API carries the token and a rejected token logs the
 * user out. This covers the many direct fetch() calls (image loading, downloads, event polling).
 */
export const installAuthFetch = () => {
  const marker = '__printchakraAuthFetch';
  if ((window as any)[marker]) return;
  (window as any)[marker] = true;

  const original = window.fetch.bind(window);
  window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    const isApi = url.startsWith(apiBase);
    const isAuthCall = url.startsWith(`${apiBase}/auth/`);

    let nextInit = init;
    if (isApi && !isAuthCall) {
      const headers = new Headers(init?.headers || (input instanceof Request ? input.headers : undefined));
      if (!headers.has('Authorization')) {
        const token = getToken();
        if (token) headers.set('Authorization', `Bearer ${token}`);
      }
      nextInit = { ...init, headers };
    }

    const response = await original(input, nextInit);
    if (isApi && !isAuthCall && response.status === 401) notifyExpired();
    return response;
  };
};

export { notifyExpired };
