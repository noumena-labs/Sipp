export * from './index'

export type ActiveNodeBackend = 'cpu' | 'cuda' | 'metal' | 'vulkan'

export declare function getActiveBackend(): ActiveNodeBackend
