export type BackendDeviceType = 'cpu' | 'gpu' | 'igpu' | 'accel' | 'unknown';

export interface BackendDeviceCapabilities {
  async: boolean;
  hostBuffer: boolean;
  bufferFromHostPtr: boolean;
  events: boolean;
}

export interface BackendDeviceInfo {
  name: string;
  description: string;
  type: BackendDeviceType;
  backendName: string;
  deviceId: string | null;
  memoryFreeBytes: number;
  memoryTotalBytes: number;
  capabilities: BackendDeviceCapabilities;
}

export interface BackendRegistryInfo {
  name: string;
  deviceCount: number;
}

export interface BackendObservability {
  profilingEnabled: boolean;
  webgpuCompiled: boolean;
  webgpuRegistered: boolean;
  webgpuDeviceCount: number;
  gpuOffloadSupported: boolean;
  engineInitialized: boolean;
  availableBackends: BackendRegistryInfo[];
  devices: BackendDeviceInfo[];
}
