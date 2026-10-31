// Resolve the probe's sole runtime package import to the real upstream source.
// Does not patch Harness source, implement a stub, or load a production package.
import { pathToFileURL } from 'node:url'
export async function resolve(specifier, context, nextResolve) {
  if (specifier === '@deepseek-ai/dsh-util-values') {
    if (!process.env.HARNESS_SOURCE) throw new Error('HARNESS_SOURCE required')
    return { url: pathToFileURL(`${process.env.HARNESS_SOURCE}/packages/util/values/src/index.ts`).href, shortCircuit: true }
  }
  return nextResolve(specifier, context)
}
