/**
 * ChisCode — main.js
 * Toast notifications · ProjectWebSocket · HTMX hooks · utilities
 */

'use strict';

/* ── Toast System ────────────────────────────────────────────────── */
const Toast = {
  _container: null,

  _init() {
    this._container = document.getElementById('toast-container');
    if (!this._container) {
      this._container = document.createElement('div');
      this._container.id = 'toast-container';
      document.body.appendChild(this._container);
    }
  },

  _icon(type) {
    const icons = {
      success: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>`,
      error:   `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`,
      info:    `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`,
      warn:    `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`,
    };
    return icons[type] || icons.info;
  },

  _colorVar(type) {
    return { success: 'var(--neon)', error: 'var(--plasma)', info: 'var(--electric)', warn: 'var(--amber)' }[type] || 'var(--electric)';
  },

  show(message, type = 'info', duration = 4500) {
    if (!this._container) this._init();

    const el = document.createElement('div');
    el.className = `toast toast--${type}`;
    el.innerHTML = `
      <span class="toast__icon" style="color:${this._colorVar(type)}">${this._icon(type)}</span>
      <span class="toast__msg">${message}</span>
      <button class="toast__close" aria-label="Dismiss">&times;</button>
    `;

    el.querySelector('.toast__close').addEventListener('click', () => this._dismiss(el));
    this._container.appendChild(el);

    if (duration > 0) {
      setTimeout(() => this._dismiss(el), duration);
    }
    return el;
  },

  _dismiss(el) {
    if (!el.isConnected) return;
    el.style.animation = 'toast-out .25s ease forwards';
    setTimeout(() => el.remove(), 250);
  },

  success: (msg, dur) => Toast.show(msg, 'success', dur),
  error:   (msg, dur) => Toast.show(msg, 'error',   dur),
  info:    (msg, dur) => Toast.show(msg, 'info',     dur),
  warn:    (msg, dur) => Toast.show(msg, 'warn',     dur),
};

window.Toast = Toast;


/* ── ProjectWebSocket ────────────────────────────────────────────── */
class ProjectWebSocket {
  constructor(projectId, handlers = {}) {
    this.projectId = projectId;
    this.on = {
      log:     handlers.onLog     || null,
      status:  handlers.onStatus  || null,
      message: handlers.onMessage || null,
      complete:handlers.onComplete|| null,
      error:   handlers.onError   || null,
    };
    this._ws            = null;
    this._ping          = null;
    this._reconnects    = 0;
    this._maxReconnects = 5;
    this._dead          = false;
  }

  connect() {
    if (this._dead) return;
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url   = `${proto}//${location.host}/projects/ws/${this.projectId}`;
    this._ws    = new WebSocket(url);

    this._ws.onopen = () => {
      this._reconnects = 0;
      this._startPing();
    };

    this._ws.onmessage = ({ data }) => {
      if (data === 'pong') return;
      try {
        const msg = JSON.parse(data);
        this._handle(msg);
      } catch { /* plain string fallback */ }
    };

    this._ws.onclose = ({ wasClean }) => {
      this._stopPing();
      if (!wasClean && !this._dead && this._reconnects < this._maxReconnects) {
        const delay = Math.min(1000 * 2 ** this._reconnects, 16000);
        this._reconnects++;
        setTimeout(() => this.connect(), delay);
      }
    };

    this._ws.onerror = () => {};
  }

  _handle(msg) {
    this.on.message?.(msg);
    switch (msg.type) {
      case 'log':
        this.on.log?.(msg.message, msg.level || 'info');
        break;
      case 'status':
        this.on.status?.(msg.status, msg.message);
        break;
      case 'complete':
        Toast.success('Generation complete!');
        this.on.complete?.(msg);
        this.on.status?.('complete', msg.message);
        break;
      case 'error':
        Toast.error(msg.message || 'Generation failed.');
        this.on.error?.(msg);
        this.on.status?.('failed', msg.message);
        break;
    }
  }

  _startPing() {
    this._ping = setInterval(() => {
      if (this._ws?.readyState === WebSocket.OPEN) this._ws.send('ping');
    }, 20000);
  }

  _stopPing() {
    clearInterval(this._ping);
    this._ping = null;
  }

  disconnect() {
    this._dead = true;
    this._stopPing();
    this._ws?.close(1000, 'done');
  }
}

window.ProjectWebSocket = ProjectWebSocket;


/* ── HTMX Global Hooks ───────────────────────────────────────────── */

// Friendly error messages from API responses
document.addEventListener('htmx:responseError', (e) => {
  const status = e.detail.xhr?.status;
  let msg = 'Something went wrong. Please try again.';
  try { msg = JSON.parse(e.detail.xhr.responseText)?.detail || msg; } catch {}

  if (status === 401) {
    msg = 'Session expired. Redirecting to login…';
    setTimeout(() => { location.href = '/login'; }, 1800);
  } else if (status === 429) {
    msg = 'Daily request limit reached. Upgrade your plan for more.';
  } else if (status === 403) {
    msg = 'You don\'t have permission to do that.';
  } else if (status === 422) {
    try {
      const errs = JSON.parse(e.detail.xhr.responseText)?.errors;
      if (errs?.length) msg = errs.map(e => `${e.field}: ${e.message}`).join(' · ');
    } catch {}
  }

  Toast.error(msg);
});

document.addEventListener('htmx:sendError', () => {
  Toast.error('Network error. Check your connection.');
});

// Server-sent toast messages via response headers
document.addEventListener('htmx:afterSwap', (e) => {
  const msg  = e.detail.xhr?.getResponseHeader('X-Toast-Message');
  const type = e.detail.xhr?.getResponseHeader('X-Toast-Type') || 'info';
  if (msg) Toast.show(msg, type);
});


/* ── Clipboard helper ────────────────────────────────────────────── */
async function copyText(text, btn) {
  try {
    await navigator.clipboard.writeText(text);
    if (btn) {
      const orig = btn.textContent;
      btn.textContent = 'Copied!';
      setTimeout(() => { btn.textContent = orig; }, 2000);
    }
    Toast.success('Copied!', 2000);
  } catch {
    Toast.error('Copy failed — please select and copy manually.');
  }
}
window.copyText = copyText;


/* ── Terminal auto-scroll ────────────────────────────────────────── */
function watchTerminal(el) {
  if (!el) return null;
  const obs = new MutationObserver(() => {
    el.scrollTop = el.scrollHeight;
  });
  obs.observe(el, { childList: true, subtree: true });
  return obs;
}
window.watchTerminal = watchTerminal;


/* ── Alpine.js data helpers ─────────────────────────────────────── */

// File tree viewer — used in project confirmation screen
function fileTreeData(files = {}) {
  return {
    files,
    active: Object.keys(files)[0] || null,
    get content() { return this.files[this.active] || ''; },
    select(path) { this.active = path; },
    get fileList() { return Object.keys(this.files).sort(); },
    get fileCount() { return Object.keys(this.files).length; },
  };
}
window.fileTreeData = fileTreeData;

// Password strength meter — used in register form
function passwordStrength(password) {
  let score = 0;
  if (password.length >= 8)              score++;
  if (/[A-Z]/.test(password))            score++;
  if (/[0-9]/.test(password))            score++;
  if (/[^A-Za-z0-9]/.test(password))     score++;
  return score; // 0–4
}
window.passwordStrength = passwordStrength;

// Status badge class helper
function statusBadge(status) {
  const map = {
    pending:               'badge-ghost',
    analyzing:             'badge-amber',
    generating:            'badge-electric',
    quality_check:         'badge-violet',
    self_healing:          'badge-amber',
    awaiting_confirmation: 'badge-amber',
    committing:            'badge-teal',
    complete:              'badge-neon',
    failed:                'badge-plasma',
    cancelled:             'badge-ghost',
  };
  return map[status] || 'badge-ghost';
}
window.statusBadge = statusBadge;


/* ── Init ────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  Toast._init();

  // Add toast-out keyframe if missing
  if (!document.querySelector('#cc-keyframes')) {
    const s = document.createElement('style');
    s.id = 'cc-keyframes';
    s.textContent = `
      @keyframes toast-out {
        from { opacity:1; transform:translateX(0); max-height:200px; margin-bottom:.6rem; }
        to   { opacity:0; transform:translateX(16px); max-height:0; margin-bottom:0; padding:0; }
      }
    `;
    document.head.appendChild(s);
  }

  // Console branding
  console.log('%cChisCode ⚡', 'color:#00e5ff;font-size:1.3rem;font-weight:800;font-family:monospace');
  console.log('%cAI-powered agent builder', 'color:#6b7f95;font-size:.85rem');
});
        
