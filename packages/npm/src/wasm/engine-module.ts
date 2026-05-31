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
  HEAP32: Int32Array;
  HEAPF32: Float32Array;
  HEAPF64: Float64Array;
  HEAPU8: Uint8Array;
  _free(ptr: number | bigint): void;
  _malloc(size: number | bigint): number | bigint;
  ccall(
    ident: string,
    returnType: string | null,
    argTypes: string[],
    args: any[],
    opts?: { async?: boolean }
  ): Promise<any> | any;
  UTF8ToString(ptr: number | bigint, maxBytesToRead?: number): string;
  addFunction(fn: Function, signature: string): number;
  removeFunction(ptr: number): void;
  // Optional host-installed hook invoked by `ce_native_yield` (see
  // `inference_runtime.cpp`). Set by the token scheduler so that one
  // runtime-event drain runs inside each JSPI yield window — this is what
  // copies token bytes into the SharedArrayBuffer token ring without
  // adding a separate macrotask source.  Untyped here because the hook lives
  // on the dynamic Emscripten Module object, not in our generated bindings.
  _ce_yield_drain?: () => void;
}
