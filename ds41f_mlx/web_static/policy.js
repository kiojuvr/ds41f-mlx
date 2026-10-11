/* Contract consumption only: no numerical defaults or sampler implementation. */
'use strict';
const GenerationPolicy = (() => {
  const parameters = ['temperature', 'top_p', 'top_k', 'min_p', 'repetition_penalty', 'presence_penalty', 'frequency_penalty', 'seed'];
  function validate(contract, values) {
    for (const name of parameters) {
      const spec = contract.parameters[name], value = values[name];
      if (name === 'seed' && value === null) continue;
      if (typeof value !== 'number' || !Number.isFinite(value) || (spec.type === 'integer' && !Number.isInteger(value))) throw new Error(name + ' must be a finite ' + (spec.type || 'number'));
      if (('minimum' in spec && value < spec.minimum) || ('maximum' in spec && value > spec.maximum) || ('exclusiveMinimum' in spec && value <= spec.exclusiveMinimum)) throw new Error(name + ' out of runtime range');
    }
    if (!contract.reasoning.efforts.includes(values.reasoning_effort)) throw new Error('Unsupported reasoning effort');
    return {...values};
  }
  function negotiate(runtime) {
    if (runtime?.status !== 'ready') throw new Error('Runtime is not READY');
    const capability = runtime.capabilities?.stateful_chat, contract = runtime.generation_policy, defaults = runtime.chat_defaults;
    if (!(capability === 'production-sessions-v1' || (capability === undefined && runtime.profile === 'standard-off'))) throw new Error('Unsupported runtime stateful_chat capability');
    const sameKeys = (value, keys) => value && Object.keys(value).sort().join() === [...keys].sort().join();
    if (!contract || contract.version !== 'generation-policy-v1' || contract.scope !== 'request' || !sameKeys(contract.parameters, parameters) || !sameKeys(contract.defaults, parameters) || !sameKeys(defaults, [...parameters, 'reasoning_effort', 'max_tokens'])) throw new Error('Unsupported generation policy contract');
    const r = contract.reasoning;
    if (!r || r.scope !== 'session' || r.stateless_scope !== 'request' || r.change !== 'reject' || r.thinking_budget_tokens !== false || r.default_thinking_mode !== false || !Array.isArray(r.efforts) || !r.efforts.includes('none')) throw new Error('Unsupported reasoning ownership contract');
    if (parameters.some(name => defaults[name] !== contract.defaults[name]) || defaults.reasoning_effort !== 'none') throw new Error('Generation defaults disagree with runtime authority');
    return validate(contract, defaults);
  }
  return {parameters, validate, negotiate};
})();
if (typeof module !== 'undefined') module.exports = GenerationPolicy;
