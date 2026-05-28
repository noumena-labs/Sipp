@echo off
call "D:\GIT_DIR\_NoumenaLabs\_CogentEngine\CogentLM\.toolchain\emsdk\emsdk_env.bat"
set PATH=D:\GIT_DIR\_NoumenaLabs\_CogentEngine\CogentLM\.toolchain\ninja;%PATH%
set EMCMAKE=emcmake.bat
set EMMAKE=emmake.bat
set RUSTFLAGS=
cargo build --release --package cogentlm-wasm --target wasm32-unknown-emscripten
