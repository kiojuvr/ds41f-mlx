/* Browser capabilities only: no model/session authority, retries or expiring leases. */
window.ChatPlatform = (() => {
  function uuid() {
    if (crypto.randomUUID) return crypto.randomUUID();
    const bytes = crypto.getRandomValues(new Uint8Array(16));
    bytes[6] = (bytes[6] & 15) | 64; bytes[8] = (bytes[8] & 63) | 128;
    const hex = [...bytes].map(x => x.toString(16).padStart(2, '0')).join('');
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
  }
  let database;
  function opened() {
    return database ||= new Promise((resolve, reject) => {
      const request = indexedDB.open('ds41f.ui-locks', 1);
      request.onupgradeneeded = () => request.result.createObjectStore('mutex');
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }
  async function withLock(name, action) {
    if (navigator.locks) return navigator.locks.request(name, {ifAvailable: true}, lock => {
      if (!lock) throw new Error('Another browser tab is using this conversation');
      return action();
    });
    // HTTP LAN origins have no Web Locks. Hold a separate IndexedDB transaction
    // until action settles. Requests keep it active across network awaits; there
    // is no time-based stealing. Navigation closes/aborts the transaction.
    // This conservatively serializes all conversations in this browser origin.
    const db = await opened();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction('mutex', 'readwrite');
      const store = transaction.objectStore('mutex');
      let entered = false, running = false, result, failure;
      const timer = setTimeout(() => { if (!entered) transaction.abort(); }, 100);
      transaction.oncomplete = () => { clearTimeout(timer); failure ? reject(failure) : resolve(result); };
      transaction.onabort = transaction.onerror = () => {
        clearTimeout(timer);
        reject(failure || transaction.error || new Error('Another browser tab is using this conversation'));
      };
      function pump() {
        const request = store.get('keepalive');
        request.onsuccess = () => {
          if (!entered) {
            entered = running = true; clearTimeout(timer);
            Promise.resolve().then(action).then(value => { result = value; running = false; }, error => { failure = error; running = false; });
          }
          if (running) pump();
        };
      }
      pump();
    });
  }
  return {uuid, withLock};
})();
