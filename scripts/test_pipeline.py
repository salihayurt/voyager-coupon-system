import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from adapters.external.data_analysis_client import DataAnalysisClient
from orchestration.workflow_engine import WorkflowEngine
from learning.state_encoder import StateEncoder
from learning.q_table import QTable
from learning.action_space import ActionSpace
from learning.reward_function import RewardCalculator
from core.rules.business_rules import RulesEngine
from agents.profitability_agent.agent import ProfitabilityAgent
from agents.conversion_agent.agent import ConversionAgent
from agents.coordinator_agent.agent import CoordinatorAgent

def create_user_info_table(user) -> Table:
    """Create a formatted table showing user information"""
    table = Table(show_header=False, box=None)
    table.add_column("Attribute", style="yellow", width=20)
    table.add_column("Value", style="white", width=30)
    
    table.add_row("🆔 User ID", str(user.user_id))
    table.add_row("🏢 Domain", user.domain.value)
    table.add_row("👥 Segment", user.segment.value.replace('_', ' ').title())
    table.add_row("🎫 One-way", "Yes" if user.is_oneway else "No")
    table.add_row("🛒 Has Basket", "Yes" if user.user_basket else "No")
    
    # Scores section
    table.add_row("", "")  # Empty row for spacing
    table.add_row("[bold]📊 Scores", "[bold]Values")
    table.add_row("  Churn Risk", f"{user.scores.churn_score:.3f}")
    table.add_row("  Activity Level", f"{user.scores.activity_score:.3f}")
    table.add_row("  Cart Abandon", f"{user.scores.cart_abandon_score:.3f}")
    table.add_row("  Price Sensitivity", f"{user.scores.price_sensitivity:.3f}")
    table.add_row("  Family Pattern", f"{user.scores.family_score:.3f}")
    
    if user.previous_domains:
        domains_str = ", ".join([d.value for d in user.previous_domains])
        table.add_row("📋 Previous Domains", domains_str)
    
    return table

def create_decision_table(decision) -> Table:
    """Create a formatted table showing the decision results"""
    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Value", style="white", width=25)
    
    # Decision metrics
    table.add_row("🎯 Final Discount", f"{decision.discount_percentage}%")
    table.add_row("📈 Expected Conversion", f"{decision.expected_conversion_rate:.1%}")
    table.add_row("💰 Expected Profit", f"{decision.expected_profit:.2f} TL")
    table.add_row("🎪 Action Type", decision.action_type.value.replace('_', ' ').title())
    
    if decision.target_domain:
        table.add_row("🎯 Target Domain", decision.target_domain.value)
    
    table.add_row("📊 Profitability Score", f"{decision.profitability_score:.3f}")
    table.add_row("🔄 Conversion Score", f"{decision.conversion_score:.3f}")
    
    return table

def main(n_users: int = 10):
    console = Console()
    
    # Header
    console.print(Panel(
        "[bold blue]🚀 VOYAGER COUPON SYSTEM - PIPELINE TEST[/bold blue]\n"
        "[dim]Testing complete workflow with real user data[/dim]",
        expand=False
    ))
    
    console.print()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        
        # Initialize components
        task = progress.add_task("🔧 Initializing system components...", total=None)
        
        try:
            # Load data
            progress.update(task, description="📂 Loading user data...")
            client = DataAnalysisClient()
            users = client.load_users(limit=n_users)
            
            if not users:
                console.print("[red]❌ No users loaded! Check your CSV file.[/red]")
                return
                
            console.print(f"[green]✅ Loaded {len(users)} users[/green]")
            
            # Initialize ML components  
            progress.update(task, description="🧠 Setting up Q-Learning components...")
            state_encoder = StateEncoder()
            q_table = QTable(state_encoder.state_space_size, ActionSpace.total_actions())
            reward_calculator = RewardCalculator()
            
            console.print(f"[green]✅ Q-Learning ready - {state_encoder.state_space_size:,} total states[/green]")
            
            # Initialize business rules
            progress.update(task, description="📋 Loading business rules...")
            rules_engine = RulesEngine()
            console.print(f"[green]✅ Business rules loaded - {len(rules_engine.rules)} rules[/green]")
            
            # Initialize agents
            progress.update(task, description="🤖 Initializing AI agents...")
            prof_agent = ProfitabilityAgent()
            conv_agent = ConversionAgent()
            coord_agent = CoordinatorAgent()
            console.print("[green]✅ AI agents ready (Profitability, Conversion, Coordinator)[/green]")
            
            # Create workflow engine
            progress.update(task, description="⚙️ Building workflow engine...")
            workflow = WorkflowEngine(
                q_table=q_table,
                state_encoder=state_encoder,
                rules_engine=rules_engine,
                profitability_agent=prof_agent,
                conversion_agent=conv_agent,
                coordinator_agent=coord_agent
            )
            console.print("[green]✅ Workflow engine ready[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Initialization failed: {e}[/red]")
            return
    
    console.print()
    console.print("[bold yellow]🔄 PROCESSING USERS[/bold yellow]")
    console.print()
    
    # Process each user
    success_count = 0
    error_count = 0
    
    for i, user in enumerate(users, 1):
        console.print(f"[bold cyan]━━━ User {i}/{len(users)} ━━━[/bold cyan]")
        
        # Show user information
        user_table = create_user_info_table(user)
        console.print(Panel(user_table, title="👤 User Profile", expand=False))
        
        # Process through workflow
        try:
            with console.status(f"[yellow]Processing user {user.user_id}...[/yellow]"):
                decision = workflow.process_user(user)
            
            # Show decision results
            decision_table = create_decision_table(decision)
            console.print(Panel(decision_table, title="🎯 Decision Results", expand=False))
            
            # Show reasoning
            if decision.reasoning:
                reasoning_text = "\n".join([f"• {reason}" for reason in decision.reasoning])
                console.print(Panel(
                    reasoning_text, 
                    title="🧠 AI Reasoning", 
                    expand=False,
                    border_style="dim"
                ))
            
            success_count += 1
            console.print("[green]✅ Processing successful[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Error processing user {user.user_id}: {e}[/red]")
            error_count += 1
        
        if i < len(users):  # Don't add extra space after last user
            console.print()
    
    # Summary
    console.print()
    console.print("[bold yellow]📊 PROCESSING SUMMARY[/bold yellow]")
    
    summary_table = Table(show_header=False)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Count", style="white")
    summary_table.add_row("✅ Successful", str(success_count))
    summary_table.add_row("❌ Errors", str(error_count))
    summary_table.add_row("📊 Success Rate", f"{(success_count/len(users)*100):.1f}%")
    
    console.print(Panel(summary_table, title="Final Results", expand=False))
    
    if success_count > 0:
        console.print("\n[bold green]🎉 Pipeline test completed successfully![/bold green]")
        console.print("[dim]The Voyager Coupon System is ready for production use.[/dim]")
    else:
        console.print("\n[bold red]❌ Pipeline test failed - please check the errors above.[/bold red]")

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    n_users = 10  # default
    if len(sys.argv) > 1:
        try:
            n_users = int(sys.argv[1])
        except ValueError:
            print("Usage: python scripts/test_pipeline.py [number_of_users]")
            sys.exit(1)
    
    print(f"Testing with {n_users} users...")
    main(n_users)
