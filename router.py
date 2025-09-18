import asyncio
import json
from pathlib import Path
from textwrap import dedent
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from agno.agent import Agent
from agno.models.aws import Claude
from agno.models.openai import OpenAIChat
from agno.tools.exa import ExaTools
from agno.tools.arxiv import ArxivTools
from agno.tools.reasoning import ReasoningTools
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.lancedb import LanceDb
from agno.db.sqlite import SqliteDb
from dotenv import load_dotenv

load_dotenv()

# Set up AWS credentials
import os
AWS_ACCESS_KEY = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_REGION = os.environ["AWS_S3_REGION"]
EXA_API_KEY = os.environ["EXA_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

class ResearchPaper(BaseModel):
    """Comprehensive structured output for research paper generation"""
    title: str = Field(
        description="Compelling, specific research paper title that clearly indicates the scope and focus"
    )
    abstract: str = Field(
        description="Comprehensive abstract (200-300 words) including background, methods, key findings, and implications"
    )
    introduction: str = Field(
        description="Detailed introduction with background context, problem statement, research questions, and paper structure"
    )
    literature_review: str = Field(
        description="Thorough literature review analyzing existing research, identifying gaps, and positioning this work"
    )
    methodology: str = Field(
        description="Detailed methodology section explaining approach, data sources, analysis methods, and limitations"
    )
    results_and_discussion: str = Field(
        description="Comprehensive results section with analysis, interpretation, and discussion of findings"
    )
    implications_and_applications: str = Field(
        description="Practical implications, real-world applications, and potential impact of the research"
    )
    limitations_and_future_work: str = Field(
        description="Honest assessment of study limitations and detailed suggestions for future research directions"
    )
    conclusion: str = Field(
        description="Strong conclusion summarizing key contributions, significance, and final thoughts"
    )
    references: List[str] = Field(
        description="Complete list of properly formatted academic references with DOIs/URLs where available"
    )
    key_insights: List[str] = Field(
        description="3-5 bullet points highlighting the most important insights and contributions"
    )
    research_gaps_identified: List[str] = Field(
        description="Specific research gaps identified during the literature review"
    )

class AdvancedResearchPipelineAgent:
    def __init__(self):
        # Initialize AWS Bedrock Claude 3.5 Sonnet
        # self.model = Claude(
        #     id="us.anthropic.claude-sonnet-4-20250514-v1:0",
        #     max_tokens=8000,
        #     aws_access_key=AWS_ACCESS_KEY,
        #     aws_region=AWS_REGION,
        #     aws_secret_key=AWS_SECRET_ACCESS_KEY
        # )
        
        self.model = OpenAIChat(api_key= OPENAI_API_KEY,
                                id= "gpt-4.1")
        
        # Configure comprehensive research tools
        self.tools = [
            # Enhanced Exa search with academic focus
            ExaTools(
                num_results=20,
                include_domains=[
                    "arxiv.org", "scholar.google.com", "researchgate.net",
                    "ieee.org", "acm.org", "springer.com", "nature.com",
                    "science.org", "plos.org", "pubmed.ncbi.nlm.nih.gov",
                    "sciencedirect.com", "jstor.org", "wiley.com",
                    "tandfonline.com", "mit.edu", "stanford.edu"
                ],
                category="research paper",
                text_length_limit=3000,
                highlights=True,
                summary=True,
                use_autoprompt=True,
                api_key=EXA_API_KEY,
            ),
            
            # ArXiv tools for academic papers
            ArxivTools(
                enable_search_arxiv=True,
                enable_read_arxiv_papers=True,
                download_dir=Path("./research_papers"),
                # max_results=15
            ),
            
            # Reasoning tools for structured thinking
            ReasoningTools(
                enable_think=True,
                enable_analyze=True,
                add_instructions=True,
                add_few_shot=True
            )
        ]
        
        # Set up local databases
        self.vector_db = LanceDb(
            table_name="research_knowledge",
            uri="./research_vectordb",
        )
        
        self.contents_db = SqliteDb(
            db_file="./research_content.db"
        )
        
        self.knowledge = Knowledge(
            vector_db=self.vector_db,
            contents_db=self.contents_db
        )
        
        self.agent_db = SqliteDb(
            db_file="./research_agent_sessions.db"
        )
        
        # Create the enhanced research agent
        self.research_agent = Agent(
            name="Advanced Research Paper Generator",
            debug_mode=True,
            model=self.model,
            tools=self.tools,
            knowledge=self.knowledge,
            db=self.agent_db,
            search_knowledge=True,
            reasoning=True,  # Enable reasoning capabilities
            instructions=self._get_comprehensive_instructions(),
            output_schema=ResearchPaper,
            markdown=True,
            add_datetime_to_context=True,
            add_name_to_context=True,
            retries=2,
            delay_between_retries=3
        )
    
    def _get_comprehensive_instructions(self) -> List[str]:
        return [
            # Core Research Methodology
            dedent("""
            **RESEARCH METHODOLOGY & APPROACH:**
            You are an expert academic researcher with deep expertise across multiple disciplines.
            Your task is to conduct comprehensive research and generate high-quality academic papers.
            
            RESEARCH PROCESS:
            1. **Initial Analysis**: Use reasoning tools to break down the topic and identify key research areas
            2. **Literature Discovery**: Search extensively using Exa for recent papers, reviews, and academic sources
            3. **ArXiv Integration**: Find relevant preprints and cutting-edge research from ArXiv
            4. **Knowledge Synthesis**: Analyze and synthesize findings from all sources
            5. **Gap Analysis**: Identify research gaps and opportunities for contribution
            6. **Original Insights**: Generate novel insights based on comprehensive analysis
            """),
            
            # Academic Writing Standards
            dedent("""
            **ACADEMIC WRITING STANDARDS:**
            - Maintain rigorous academic tone and scholarly language throughout
            - Use precise, technical terminology appropriate to the field
            - Ensure logical flow and coherent argumentation
            - Support ALL claims with evidence and proper citations
            - Provide balanced, objective analysis of different perspectives
            - Include quantitative data and statistics where relevant
            - Use active voice and clear, concise sentences
            """),
            
            # Citation and Evidence Requirements
            dedent("""
            **CITATION & EVIDENCE REQUIREMENTS:**
            - Include exact sentence-level citations in format: "Statement here (Author et al., Year)"
            - Ensure every factual claim is supported by credible sources
            - Use recent sources (prefer last 5 years, include seminal older works when relevant)
            - Include DOI, ArXiv ID, or URL for all references when available
            - Properly attribute all ideas, methods, and findings to original authors
            - Use a mix of primary research papers, review articles, and authoritative sources
            """),
            
            # Structure and Organization
            dedent("""
            **PAPER STRUCTURE & ORGANIZATION:**
            - Create compelling section headings that guide the reader
            - Ensure smooth transitions between sections
            - Build arguments progressively from introduction to conclusion
            - Use subsections where appropriate for complex topics
            - Include clear topic sentences and paragraph structure
            - Maintain consistent terminology throughout the paper
            """),
            
            # Quality Assurance
            dedent("""
            **QUALITY ASSURANCE:**
            - Verify all facts and figures against multiple sources
            - Ensure methodological rigor in analysis and interpretation
            - Address potential counterarguments and limitations honestly
            - Provide practical implications and real-world applications
            - Suggest specific, actionable future research directions
            - Maintain objectivity while highlighting significance of findings
            """),
            
            # Reasoning Integration
            dedent("""
            **REASONING INTEGRATION:**
            - Use reasoning tools to think through complex problems step-by-step
            - Analyze relationships between different research findings
            - Evaluate the strength of evidence for different claims
            - Consider multiple perspectives and potential biases
            - Synthesize information from diverse sources into coherent insights
            - Apply critical thinking to identify research gaps and opportunities
            """)
        ]
    
    def format_research_paper(self, paper: ResearchPaper) -> str:
        """Format the research paper with proper academic structure and styling"""
        
        formatted_paper = f"""
# {paper.title}

## Abstract

{paper.abstract}

**Keywords:** {', '.join(paper.key_insights[:5])}

---

## 1. Introduction

{paper.introduction}

## 2. Literature Review

{paper.literature_review}

### 2.1 Research Gaps Identified

{chr(10).join(f"• {gap}" for gap in paper.research_gaps_identified)}

## 3. Methodology

{paper.methodology}

## 4. Results and Discussion

{paper.results_and_discussion}

## 5. Implications and Applications

{paper.implications_and_applications}

## 6. Limitations and Future Work

{paper.limitations_and_future_work}

## 7. Conclusion

{paper.conclusion}

### Key Insights

{chr(10).join(f"• {insight}" for insight in paper.key_insights)}

## References

{chr(10).join(f"{i+1}. {ref}" for i, ref in enumerate(paper.references))}

---

*Generated on {datetime.now().strftime("%B %d, %Y")} using Advanced Research Pipeline*
*Total References: {len(paper.references)}*
"""
        return formatted_paper
    
    async def generate_research_paper(self, topic: str, focus_areas: Optional[List[str]] = None) -> str:
        """Generate a comprehensive research paper with enhanced reasoning and formatting"""
        
        # Construct detailed research prompt
        focus_context = ""
        if focus_areas:
            focus_context = f"\n\nSPECIFIC FOCUS AREAS:\n" + "\n".join(f"- {area}" for area in focus_areas)
        
        research_prompt = f"""
        **RESEARCH ASSIGNMENT:** Generate a comprehensive, high-quality research paper on the topic: "{topic}"
        {focus_context}
        
        **DETAILED REQUIREMENTS:**
        
        1. **COMPREHENSIVE RESEARCH PROCESS:**
           - Begin by using reasoning tools to analyze the topic and plan your research approach
           - Conduct extensive literature search using Exa tools across academic databases
           - Search ArXiv for the latest preprints and cutting-edge research
           - Analyze and synthesize findings from all discovered sources
           - Identify research gaps and opportunities for original contribution
        
        2. **PAPER SPECIFICATIONS:**
           - Target length: 2500-3000 words (approximately 2 full pages when printed)
           - Academic rigor equivalent to journal publication standards
           - Comprehensive coverage of the topic with depth and breadth
           - Original insights and analysis based on synthesized research
        
        3. **CONTENT REQUIREMENTS:**
           - **Abstract:** 200-300 words summarizing the entire paper
           - **Introduction:** Establish context, problem statement, and research objectives
           - **Literature Review:** Comprehensive analysis of existing research with critical evaluation
           - **Methodology:** Detailed explanation of research approach and analysis methods
           - **Results & Discussion:** In-depth analysis of findings with interpretation
           - **Implications:** Practical applications and real-world significance
           - **Limitations:** Honest assessment of study constraints and boundaries
           - **Future Work:** Specific, actionable research directions
           - **Conclusion:** Strong synthesis of key contributions and significance
        
        4. **CITATION STANDARDS:**
           - Include exact sentence-level citations for ALL factual claims
           - Use format: "Statement here (Author et al., Year)"
           - Minimum 15-20 high-quality academic references
           - Include DOI/ArXiv ID/URL for all references
           - Prioritize recent sources (last 5 years) while including seminal works
        
        5. **QUALITY STANDARDS:**
           - Maintain scholarly tone and academic rigor throughout
           - Ensure logical flow and coherent argumentation
           - Support all claims with credible evidence
           - Provide balanced, objective analysis
           - Include quantitative data where relevant
           - Address potential limitations and counterarguments
        
        **RESEARCH FOCUS:** Ensure the paper makes a meaningful contribution to the field by:
        - Identifying and analyzing current research gaps
        - Synthesizing information from diverse sources
        - Providing original insights and analysis
        - Suggesting practical applications and implications
        - Proposing specific future research directions
        
        **OUTPUT FORMAT:** Structure your response using the ResearchPaper schema with all required fields properly filled.
        """
        
        # Generate the paper using the agent
        print(f"🔬 Starting comprehensive research on: {topic}")
        print("📚 Searching academic databases and ArXiv...")
        print("🧠 Applying reasoning and analysis...")
        print("✍️  Generating research paper...")
        
        response = await self.research_agent.arun(research_prompt)
        
        # Format the paper for display
        if isinstance(response.content, ResearchPaper):
            formatted_paper = self.format_research_paper(response.content)
            return formatted_paper
        else:
            return str(response.content)
    
    def generate_research_paper_sync(self, topic: str, focus_areas: Optional[List[str]] = None) -> str:
        """Synchronous version of research paper generation"""
        
        focus_context = ""
        if focus_areas:
            focus_context = f"\n\nSPECIFIC FOCUS AREAS:\n" + "\n".join(f"- {area}" for area in focus_areas)
        
        research_prompt = f"""
        **RESEARCH ASSIGNMENT:** Generate a comprehensive, high-quality research paper on: "{topic}"
        {focus_context}
        
        **COMPREHENSIVE RESEARCH PROCESS:**
        1. Use reasoning tools to analyze the topic and develop research strategy
        2. Conduct extensive academic literature search using Exa tools
        3. Search ArXiv for latest preprints and cutting-edge research
        4. Synthesize findings from all sources into coherent analysis
        5. Generate original insights and identify research contributions
        
        **PAPER REQUIREMENTS:**
        - 2500-3000 words of high-quality academic content
        - Comprehensive literature review with critical analysis
        - Original insights and synthesis of existing research
        - Exact sentence-level citations: "Statement (Author et al., Year)"
        - Minimum 15-20 academic references with DOI/URLs
        - Professional academic writing style and structure
        - Practical implications and future research directions
        
        **QUALITY STANDARDS:**
        - Journal-publication level academic rigor
        - Evidence-based claims with proper attribution
        - Balanced, objective analysis of multiple perspectives
        - Clear identification of research gaps and contributions
        - Actionable recommendations for future work
        
        Generate a paper that makes a meaningful contribution to the field through comprehensive research and original analysis.
        """
        
        print(f"🔬 Generating research paper on: {topic}")
        print("📊 This may take several minutes for comprehensive research...")
        
        response = self.research_agent.run(research_prompt)
        
        # Format the paper for display
        if isinstance(response.content, ResearchPaper):
            formatted_paper = self.format_research_paper(response.content)
            return formatted_paper
        else:
            return str(response.content)
    
    def save_paper(self, paper_content: str, topic: str, output_dir: str = "./generated_papers") -> str:
        """Save the generated paper to a file with proper naming"""
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Generate filename
        safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_topic = safe_topic.replace(' ', '_').lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_paper_{safe_topic}_{timestamp}.md"
        filepath = Path(output_dir) / filename
        
        # Save the paper
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(paper_content)
        
        print(f"📄 Paper saved to: {filepath}")
        return str(filepath)

# Usage Examples and Main Functions
async def main():
    """Main function demonstrating the research pipeline"""
    
    # Initialize the advanced pipeline
    pipeline = AdvancedResearchPipelineAgent()
    
    # Example topics with focus areas
    research_topics = [
        {
            "topic": "Large Language Models in Scientific Research",
            "focus_areas": [
                "Automated literature review and synthesis",
                "Hypothesis generation and experimental design",
                "Scientific writing and peer review assistance",
                "Ethical considerations and limitations"
            ]
        },
        {
            "topic": "Quantum Computing Applications in Machine Learning",
            "focus_areas": [
                "Quantum machine learning algorithms",
                "Near-term quantum advantage",
                "Hybrid classical-quantum systems",
                "Current limitations and future prospects"
            ]
        }
    ]
    
    # Generate research papers
    for research_config in research_topics:
        topic = research_config["topic"]
        focus_areas = research_config["focus_areas"]
        
        print(f"\n{'='*80}")
        print(f"🎯 RESEARCH TOPIC: {topic}")
        print(f"🔍 FOCUS AREAS: {', '.join(focus_areas)}")
        print(f"{'='*80}")
        
        try:
            # Generate the paper
            paper_content = await pipeline.generate_research_paper(topic, focus_areas)
            
            # Save the paper
            filepath = pipeline.save_paper(paper_content, topic)
            
            print(f"\n✅ Research paper generated successfully!")
            print(f"📁 Saved to: {filepath}")
            print(f"📊 Local databases updated with research findings")
            
            # Display first part of the paper
            print(f"\n{'='*60}")
            print("📖 PAPER PREVIEW:")
            print(f"{'='*60}")
            print(paper_content[:1500] + "..." if len(paper_content) > 1500 else paper_content)
            
        except Exception as e:
            print(f"❌ Error generating paper for '{topic}': {str(e)}")
            continue

def generate_single_paper():
    """Synchronous function to generate a single research paper"""
    
    pipeline = AdvancedResearchPipelineAgent()
    
    # Get topic from user or use default
    topic = input("Enter research topic (or press Enter for default): ").strip()
    if not topic:
        topic = "Large Language Models in Scientific Research"
    
    focus_areas = [
                "Automated literature review and synthesis",
                "Hypothesis generation and experimental design",
                "Scientific writing and peer review assistance",
                "Ethical considerations and limitations"
    ]
    
    print(f"\n🔬 Generating research paper on: '{topic}'")
    print("⏳ This process may take 3-5 minutes for comprehensive research...")
    
    try:
        # Generate paper
        paper_content = pipeline.generate_research_paper_sync(topic, focus_areas)
        
        # Save paper
        filepath = pipeline.save_paper(paper_content, topic)
        
        print(f"\n✅ SUCCESS! Research paper generated and saved.")
        print(f"📁 File location: {filepath}")
        print(f"📚 Knowledge base updated with research findings")
        
        return paper_content
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

if __name__ == "__main__":
    print("🚀 Advanced Research Paper Generation Pipeline")
    print("=" * 60)
    
    # Choose execution mode
    mode = input("Choose mode - (1) Async multiple papers, (2) Single paper: ").strip()
    
    if mode == "1":
        # Run async version for multiple papers
        asyncio.run(main())
    else:
        # Run sync version for single paper
        generate_single_paper()
    
    print("\n🎉 Research pipeline execution completed!")
    print("📊 Check the generated_papers/ directory for your research papers")
    print("💾 Local databases contain all research findings for future use")