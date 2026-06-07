export * from './index'

/** Native backend selected by the Node package loader. */
export type ActiveNodeBackend = 'cpu' | 'cuda' | 'metal' | 'vulkan'

/** Return the backend selected for the currently loaded native binding. */
export declare function getActiveBackend(): ActiveNodeBackend
