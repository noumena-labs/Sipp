import { CogentEngine } from '../dist/esm/index.js';

async function run() {
  console.log("Initializing CogentEngine...");
  
  // Note: For node.js environment, the WebAssembly module might require 
  // special flags like --experimental-wasm-threads
  const engine = new CogentEngine();
  
  // Example dummy loop to show the API
  try {
    // 1. Initialize engine
    await engine.initModule();
    console.log("WASM Module loaded successfully!");

    // 2. Load model (Assume there is a local tiny model for testing)
    // const modelBuffer = fs.readFileSync('./models/tinyllama.gguf');
    // const path = engine.loadModelFromBuffer(modelBuffer, 'model.gguf');
    
    // 3. Init Engine
    // await engine.initEngine(path);
    // console.log("Engine initialized with model!");

    // 4. Test Submitting State
    /*
    engine.submitAgentState("agent_0", {
      name: "Assistant",
      persona: "A helpful AI assistant",
    });

    // 5. Generate prompt
    console.log("Generating prompt...");
    const response = await engine.prompt("chat", "Hello Assistant, what is your name?");
    console.log("Response:", response);
    */
   
    console.log("Example completed. Start developing by loading a real GGUF model!");
  } catch (err) {
    console.error("Error running test:", err);
  }
}

run();
