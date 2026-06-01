mergeInto(LibraryManager.library, {
  ce_native_yield__sig: 'v',
  ce_native_yield: function () {
    var fn = Module && Module._ce_yield_drain;
    if (typeof fn === 'function') {
      fn();
    }
  },
});
