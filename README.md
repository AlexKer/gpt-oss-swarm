# GPT-OSS Swarm Mode

A powerful async implementation that generates multiple AI responses in parallel, scores them using an LLM judge, and synthesizes the best parts into one comprehensive answer.

## 🚀 What it does

1. **Generate**: Creates 10 responses concurrently using async
2. **Score**: Uses LLM judge to rate each response (1-10)  
3. **Synthesize**: Combines the top candidates into one final answer

Perfect for complex analysis, creative tasks, and getting multiple perspectives on challenging questions.

## 📦 Installation

```bash
pip install openai asyncio
```

## 🔑 Setup

### 1. Get Baseten API Key

1. Sign up at [Baseten](https://www.baseten.co/)
2. Get your API key from the dashboard
3. Set the environment variable:

```bash
# Option 1: Export in terminal
export BASETEN_API_KEY="your_api_key_here"

# Option 2: Add to your .bashrc/.zshrc
echo 'export BASETEN_API_KEY="your_api_key_here"' >> ~/.zshrc
source ~/.zshrc

# Option 3: Create .env file (if using python-dotenv)
echo "BASETEN_API_KEY=your_api_key_here" > .env
```

### 2. Verify Setup

```python
import os
print(os.getenv("BASETEN_API_KEY"))  # Should print your key
```

## 🎯 Usage

### Basic Usage

```python
import asyncio
from swarm_mode import swarm_generate

async def main():
    prompt = "How should companies respond to AI disruption?"
    result = await swarm_generate(prompt, n_candidates=10)
    print(result["final_answer"])

asyncio.run(main())
```

### Advanced Usage

```python
from swarm_mode import SwarmMode, SwarmConfig

# Custom configuration
config = SwarmConfig(
    candidate_temperature=0.8,  # More creative candidates
    synthesis_temperature=0.2,  # More focused synthesis
    top_candidates=5,           # Use top 5 out of 10
    max_retries=3
)

async with SwarmMode(config) as swarm:
    result = await swarm.generate(prompt, n_candidates=10)
    
print(f"Success rate: {result['success_rate']:.1%}")
print(f"Top scores: {result['scores']}")
print(f"Final answer: {result['final_answer']}")
```

## 🎬 Example Output

```
🔥 Generating 10 candidates in parallel...
🚀 Starting candidate 1...
🚀 Starting candidate 2...
...
🤖 AI swarm finished thinking ✨ (2.3s)

✅ Candidate 3 completed!
   Preview: OpenAI's release of GPT OSS 120B represents...

✨ Generated 10/10 candidates (100%)

🎯 Scoring candidates and selecting top 5...
📊 All scores: [8.5, 7.2, 6.8, 6.5, 6.1, 5.9, 5.4, 4.7, 4.2, 3.8]
🏆 Selected top 5 candidates with scores: [8.5, 7.2, 6.8, 6.5, 6.1]

🧠 Synthesizing final answer from top 5 candidates...
🎉 Synthesis complete! (3.1s)

============================================================
FINAL ANSWER:
============================================================
[Comprehensive synthesized analysis]
```

## ⚙️ Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"openai/gpt-oss-120b"` | The model to use |
| `max_tokens` | `2000` | Maximum tokens per response |
| `candidate_temperature` | `0.8` | Temperature for candidate generation (higher = more creative) |
| `synthesis_temperature` | `0.2` | Temperature for synthesis (lower = more focused) |
| `max_retries` | `3` | Number of retries on API failure |
| `top_candidates` | `3` | Number of top candidates to synthesize |

## 🔧 Troubleshooting

### API Key Issues
```bash
# Check if key is set
echo $BASETEN_API_KEY

# If empty, set it:
export BASETEN_API_KEY="your_key_here"
```

### Import Issues
```bash
# Make sure you're in the right directory
ls swarm_mode.py  # Should exist

# Run from the same directory as swarm_mode.py
python3 -c "from swarm_mode import swarm_generate; print('✅ Import successful')"
```

### Rate Limiting
If you hit rate limits, the code will automatically retry with exponential backoff.

## 🎯 Use Cases

- **Strategic Analysis**: Get multiple perspectives on business decisions
- **Creative Writing**: Generate diverse creative content and combine the best parts  
- **Research Questions**: Comprehensive analysis from multiple angles
- **Problem Solving**: Different approaches to complex problems

## 🚀 Performance

- **Concurrent Generation**: 10 responses in ~2-3 seconds (vs 20+ seconds sequential)
- **Smart Selection**: Only the best responses are used for synthesis
- **Async Design**: Non-blocking, efficient resource usage

## 📄 License

MIT License - feel free to use and modify! 