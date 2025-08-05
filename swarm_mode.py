# GPT Oss Swarm Mode - AsyncIO-based multi-candidate generation, scoring, and synthesis
import asyncio
import logging
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

# Configuration
@dataclass
class SwarmConfig:
    model: str = "openai/gpt-oss-120b"
    max_tokens: int = 2000
    candidate_temperature: float = 0.8 # Make candidates more creative
    synthesis_temperature: float = 0.2 # Make synthesis more stable
    max_retries: int = 3
    top_candidates: int = 3
    


@dataclass
class CompletionResult:
    """Result of a completion with metadata"""
    content: str
    success: bool
    error: Optional[str] = None

class ScoreExtractor:
    """Handles extracting numeric scores from text responses"""
    
    PATTERNS = [
        r"^(\d+(?:\.\d+)?)$",                     # Just a number by itself
        r"^\s*(\d+(?:\.\d+)?)\s*$",               # Number with whitespace
        r"(?:score|rating)[:\s]*(\d+(?:\.\d+)?)", # "Score: 7" or "Rating: 7"
        r"(\d+(?:\.\d+)?)\s*/\s*10",              # "8/10" or "8 / 10"
        r"\b(\d+(?:\.\d+)?)\b",                   # Any standalone number (last resort)
    ]
    
    @classmethod
    def extract(cls, response_text: Optional[str]) -> float:
        """Extract numeric score from response text."""
        if not response_text:
            return 5.0
        
        # Try specific patterns first
        for i, pattern in enumerate(cls.PATTERNS):
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            if matches:
                scores = [float(m) for m in matches if 0 <= float(m) <= 10]
                if scores:
                    return max(scores)
        
        # Fallback: look for numbers, but be more selective
        # First try to find numbers that look like scores (not in prompts)
        lines = response_text.split('\n')
        for line in lines:
            # Skip lines that look like they're repeating the prompt
            if 'evaluate' in line.lower() or 'user asks' in line.lower():
                continue
            numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', line)
            valid_scores = [float(n) for n in numbers if 1.0 <= float(n) <= 10.0]
            if valid_scores:
                return valid_scores[0]
        
        return 5.0

class SwarmMode:
    """Clean implementation of swarm mode for LLM generation"""
    
    def __init__(self, config: SwarmConfig = None):
        self.config = config or SwarmConfig()
        self.logger = self.setup_logger()
        self.client: Optional[AsyncOpenAI] = None
    
    def setup_logger(self) -> logging.Logger:
        """Setup logger with appropriate level"""
        logger = logging.getLogger("swarm_mode")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.client = AsyncOpenAI(
            api_key=os.getenv("BASETEN_API_KEY"),
            base_url="https://inference.baseten.co/v1"
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.client:
            await self.client.close()
    
    async def generate_single_candidate(self, prompt: str, index: int) -> CompletionResult:
        """Generate a single completion with retry logic."""
        print(f"🚀 Starting candidate {index + 1}...")
        
        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.config.model,
                    temperature=self.config.candidate_temperature,
                    max_tokens=self.config.max_tokens,
                    stream=False
                )
                
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty response from API")
                
                # Show preview of the response
                preview = content[:200] + "..." if len(content) > 200 else content
                print(f"✅ Candidate {index + 1} completed!")
                print(f"   Preview: {preview}\n")
                return CompletionResult(content=content, success=True)
                
            except Exception as e:
                print(f"⚠️  Candidate {index + 1} attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    return CompletionResult(content="", success=False, error=str(e))
                await asyncio.sleep(2 ** attempt)
        
        return CompletionResult(content="", success=False, error="Max retries exceeded")
    
    async def score_candidate(self, candidate: str, original_prompt: str) -> float:
        """Score a single candidate response."""
        scoring_prompt = f"""Evaluate this response and give it a score from 1-10.

Response to evaluate:
{candidate[:500]}...

Rate based on:
- Quality of insights (1-4 points)
- Depth of analysis (1-3 points) 
- Clarity and structure (1-3 points)

Be critical. Most responses should score 4-7. Only exceptional responses deserve 8-10.

Score (just the number):"""
        
        try:
            response = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": scoring_prompt}],
                model=self.config.model,
                temperature=0.2,
                max_tokens=100,
                stream=False
            )
            
            content = response.choices[0].message.content
            reasoning = getattr(response.choices[0].message, 'reasoning', None)
            score_text = content or reasoning
            
            return ScoreExtractor.extract(score_text)
            
        except Exception as e:
            self.logger.warning(f"Error scoring candidate: {e}")
            return 5.0
    
    async def score_candidates(self, candidates: List[str], original_prompt: str) -> List[tuple]:
        """Score all candidates and return top ones."""
        # Score all candidates concurrently
        tasks = [self.score_candidate(candidate, original_prompt) for candidate in candidates]
        scores = await asyncio.gather(*tasks)
        
        # Show all scores for transparency
        print(f"📊 All scores: {sorted(scores, reverse=True)}")
        
        # Combine and sort
        candidate_scores = list(zip(candidates, scores))
        sorted_scores = sorted(candidate_scores, key=lambda x: x[1], reverse=True)
        
        # Return top N
        top_n = sorted_scores[:self.config.top_candidates]
        
        print(f"🏆 Selected top {len(top_n)} candidates with scores: {[s for _, s in top_n]}")
        return top_n
    
    def create_synthesis_prompt(self, candidates: List[str]) -> List[Dict[str, str]]:
        """Create synthesis prompt for combining responses."""
        candidate_blocks = [
            f"[RESPONSE_{i+1}]\n{candidate}\n[/RESPONSE_{i+1}]"
            for i, candidate in enumerate(candidates)
        ]
        
        return [
            {
                "role": "system",
                "content": (
                    "You are an expert synthesizer. Create a comprehensive answer that combines "
                    "the best elements from multiple responses. Focus on accuracy, completeness, "
                    "and clarity while eliminating redundancy and contradictions."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Below are {len(candidates)} responses to the same question:\n\n"
                    f"{chr(10).join(candidate_blocks)}\n\n"
                    f"Create the single best possible answer by combining their strengths."
                )
            }
        ]
    
    async def generate(self, prompt: str, n_candidates: int = 5) -> Dict[str, Any]:
        """
        Generate multiple candidates, score them, and synthesize the best ones.
        
        Args:
            prompt: Input prompt to analyze
            n_candidates: Number of candidates to generate
            
        Returns:
            Dict with final_answer, candidates, scores, and metadata
        """
        if not self.client:
            raise RuntimeError("SwarmMode must be used as async context manager")
        
        self.logger.info(f"Generating {n_candidates} candidates")
        
        # Generate candidates concurrently
        print(f"\n🔥 Generating {n_candidates} candidates in parallel...")
        tasks = [self.generate_single_candidate(prompt, i) for i in range(n_candidates)]
        
        # Show animated loading after launching
        import time
        start_time = time.time()
        await asyncio.sleep(0.1)  # Let the launch messages appear first
        
        print("\n🤖 AI swarm is thinking", end="", flush=True)
        dots_count = 0
        
        # Create a simple animation task
        async def animate_loading():
            nonlocal dots_count
            while True:
                await asyncio.sleep(0.5)
                dots_count = (dots_count + 1) % 4
                print(f"\r🤖 AI swarm is thinking{'.' * dots_count}{' ' * (3 - dots_count)}", end="", flush=True)
        
        # Start animation and generation concurrently
        animation_task = asyncio.create_task(animate_loading())
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Stop animation
        animation_task.cancel()
        elapsed = time.time() - start_time
        print(f"\r🤖 AI swarm finished thinking ✨ ({elapsed:.1f}s)")
        
        # Extract successful candidates
        candidates = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Generation error: {result}")
                continue
            if result.success:
                candidates.append(result.content)
        
        if not candidates:
            raise RuntimeError("No candidates generated successfully")
        
        success_rate = len(candidates) / n_candidates
        print(f"\n✨ Generated {len(candidates)}/{n_candidates} candidates ({success_rate:.1%})")
        
        # Score and select top candidates
        print(f"\n🎯 Scoring candidates and selecting top {self.config.top_candidates}...")
        
        start_time = time.time()
        scored_candidates = await self.score_candidates(candidates, prompt)
        elapsed = time.time() - start_time
        top_candidates = [candidate for candidate, _ in scored_candidates]
        scores = [score for _, score in scored_candidates]
        
        # Synthesize final answer
        print(f"\n🧠 Synthesizing final answer from top {len(top_candidates)} candidates...")
        messages = self.create_synthesis_prompt(top_candidates)
        
        start_time = time.time()
        response = await self.client.chat.completions.create(
            messages=messages,
            model=self.config.model,
            temperature=self.config.synthesis_temperature,
            max_tokens=self.config.max_tokens,
            stream=False
        )
        
        final_answer = response.choices[0].message.content
        elapsed = time.time() - start_time
        print(f"🎉 Synthesis complete! ({elapsed:.1f}s)\n")
        
        return {
            "final_answer": final_answer,
            "candidates": candidates,
            "scores": scores,
            "success_rate": success_rate
        }

# Convenience function
async def swarm_generate(
    prompt: str, 
    n_candidates: int = 5, 
    config: SwarmConfig = None
) -> Dict[str, Any]:
    """Convenience function for one-off swarm generation."""
    async with SwarmMode(config) as swarm:
        return await swarm.generate(prompt, n_candidates)

if __name__ == "__main__":
    # Market analysis example
    async def test():
        prompt = """
        OpenAI just released GPT OSS 120B, an open-source model that matches GPT-4o performance.
        How should major tech companies respond to this open-source disruption?
        Provide strategic recommendations for OpenAI/Microsoft, Google/Meta, and the startup ecosystem.
        """
        
        config = SwarmConfig(top_candidates=5)
        result = await swarm_generate(prompt, n_candidates=10, config=config)
        
        print("=" * 60)
        print("FINAL ANSWER:")
        print("=" * 60)
        print(result["final_answer"])
        print(f"📊 Top scores: {result['scores']}")
    
    asyncio.run(test())