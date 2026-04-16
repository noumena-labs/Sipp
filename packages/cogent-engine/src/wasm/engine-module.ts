export interface EmscriptenFs {
  analyzePath(path: string): { exists: boolean };
  mkdir(path: string): void;
  writeFile(path: string, data: Uint8Array): void;
  unlink(path: string): void;
  mount(type: any, opts: any, mountpoint: string): void;
  unmount(mountpoint: string): void;
}

export interface EngineModule {
  FS: EmscriptenFs;
  WORKERFS: any;
  HEAP32: Int32Array;
  HEAPF64: Float64Array;
  HEAPU8: Uint8Array;
  _free(ptr: number | bigint): void;
  _malloc(size: number | bigint): number | bigint;
  addFunction(func: (...args: any[]) => any, signature: string): number | bigint;
  ccall(
    ident: string,
    returnType: string | null,
    argTypes: string[],
    args: any[],
    opts?: { async?: boolean }
  ): Promise<any> | any;
  removeFunction(ptr: number | bigint): void;
  UTF8ToString(ptr: number | bigint, maxBytesToRead?: number): string;
}
