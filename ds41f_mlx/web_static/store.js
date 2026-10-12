/* Tab-lifetime ordinary history and uncertainty journals, never an archive.
 * Losing the page never authorizes request/effect resend or native reconstruction.
 */
window.ChatStore = (() => {
  const sessions = new Map();
  function table(name) {
    if (name !== 'sessions') throw new Error('Persistent conversation archives are not supported');
    return sessions;
  }
  return {
    metadata: async name => [...table(name).values()].map(s => ({id: s.id, title: s.title})),
    all: async name => [...table(name).values()],
    get: async (name, id) => table(name).get(id),
    put: async (name, value) => { table(name).set(value.id, value); },
    remove: async (name, id) => { table(name).delete(id); },
  };
})();
