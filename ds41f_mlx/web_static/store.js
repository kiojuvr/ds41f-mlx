/* Application history only. No KV, token frontier reconstruction or model truth. */
window.ChatStore = (() => {
  const opened = new Promise((resolve, reject) => {
    const request = indexedDB.open('ds41f.application', 1);
    request.onupgradeneeded = () => {
      for (const name of ['sessions', 'saves']) request.result.createObjectStore(name, {keyPath: 'id'});
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  async function operate(name, mode, method, value) {
    const db = await opened;
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(name, mode);
      const request = transaction.objectStore(name)[method](value);
      transaction.oncomplete = () => resolve(request.result);
      transaction.onabort = () => reject(transaction.error || request.error);
      transaction.onerror = () => reject(transaction.error || request.error);
    });
  }
  return {
    all: name => operate(name, 'readonly', 'getAll'),
    get: (name, id) => operate(name, 'readonly', 'get', id),
    put: (name, value) => operate(name, 'readwrite', 'put', value),
    remove: (name, id) => operate(name, 'readwrite', 'delete', id),
  };
})();
