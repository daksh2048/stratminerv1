"""
Analyze all backtest results across strategies and symbols
Run this after all 5 batches are complete
"""
import pandas as pd
import glob
import os

def main():
    results = []
    
    # Find all trade CSV files
    csv_files = glob.glob("backtests/trades_*.csv")
    
    if not csv_files:
        print("ERROR: No backtest CSV files found in backtests/ folder")
        print("Make sure backtests have finished running")
        return
    
    print(f"Found {len(csv_files)} backtest files")
    print("Analyzing...\n")
    
    for csv_file in csv_files:
        # Parse filename: trades_STRATEGY_SYMBOL_5m.csv
        filename = os.path.basename(csv_file)
        parts = filename.replace("trades_", "").replace("_5m.csv", "").split("_")
        
        if len(parts) < 2:
            continue
        
        # Handle multi-word strategies
        if "mean" in filename and "reversion" in filename:
            strategy = "rsi_mean_reversion"
            symbol = parts[-1]
        elif "squeeze" in filename or "bollinger" in filename:
            strategy = "bollinger_squeeze"
            symbol = parts[-1]
        elif "crossover" in filename:
            strategy = "ma_crossover"
            symbol = parts[-1]
        elif "breakout" in filename and "sr" in filename:
            strategy = "sr_breakout"
            symbol = parts[-1]
        elif "fill" in filename or "gap" in filename:
            strategy = "gap_fill"
            symbol = parts[-1]
        else:
            # Fallback for old strategies
            strategy = parts[0]
            symbol = parts[1] if len(parts) > 1 else "UNKNOWN"
        
        try:
            df = pd.read_csv(csv_file)
            
            if len(df) == 0:
                print(f"  WARNING: {filename} has no trades")
                continue
            
            total_trades = len(df)
            wins = len(df[df['result'] == 'win'])
            win_rate = wins / total_trades if total_trades > 0 else 0
            avg_R = df['R'].mean()
            final_balance = df['balance'].iloc[-1]
            total_return = ((final_balance - 10000) / 10000) * 100
            
            results.append({
                'Strategy': strategy,
                'Symbol': symbol,
                'Trades': total_trades,
                'WR': f"{win_rate*100:.1f}%",
                'Avg_R': f"{avg_R:+.3f}",
                'Return': f"{total_return:+.1f}%",
                'Return_num': total_return,  # For sorting
                'Final': f"${final_balance:.0f}"
            })
        except Exception as e:
            print(f"  ERROR processing {filename}: {e}")
            continue
    
    if not results:
        print("ERROR: No valid results found")
        return
    
    # Create DataFrame and sort
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Return_num', ascending=False)
    
    # Display full results
    print("=" * 120)
    print("COMPLETE BACKTEST RESULTS - ALL STRATEGIES & SYMBOLS")
    print("=" * 120)
    print(results_df[['Strategy', 'Symbol', 'Trades', 'WR', 'Avg_R', 'Return', 'Final']].to_string(index=False))
    
    # Strategy summary
    print("\n" + "=" * 120)
    print("SUMMARY BY STRATEGY:")
    print("-" * 120)
    
    for strat in sorted(results_df['Strategy'].unique()):
        strat_df = results_df[results_df['Strategy'] == strat]
        profitable = len(strat_df[strat_df['Return_num'] > 0])
        total = len(strat_df)
        avg_return = strat_df['Return_num'].mean()
        max_return = strat_df['Return_num'].max()
        min_return = strat_df['Return_num'].min()
        
        print(f"{strat:25} | Symbols: {total:2} | Profitable: {profitable:2}/{total:2} ({profitable/total*100:5.1f}%) | " +
              f"Avg: {avg_return:+6.1f}% | Best: {max_return:+6.1f}% | Worst: {min_return:+6.1f}%")
    
    # Top performers
    print("\n" + "=" * 120)
    print("TOP 20 STRATEGY-SYMBOL PAIRS:")
    print("-" * 120)
    top20 = results_df.head(20)
    print(top20[['Strategy', 'Symbol', 'Trades', 'WR', 'Avg_R', 'Return']].to_string(index=False))
    
    # Winners (>15% return)
    print("\n" + "=" * 120)
    print("EXCELLENT PERFORMERS (>15% return):")
    print("-" * 120)
    winners = results_df[results_df['Return_num'] > 15]
    if len(winners) > 0:
        print(winners[['Strategy', 'Symbol', 'Trades', 'WR', 'Avg_R', 'Return']].to_string(index=False))
    else:
        print("None found")
    
    # Losers (<-5% return)
    print("\n" + "=" * 120)
    print("POOR PERFORMERS (<-5% return):")
    print("-" * 120)
    losers = results_df[results_df['Return_num'] < -5]
    if len(losers) > 0:
        print(losers[['Strategy', 'Symbol', 'Trades', 'WR', 'Avg_R', 'Return']].to_string(index=False))
    else:
        print("None found")
    
    # Save to CSV for further analysis
    output_file = "backtests/COMPLETE_ANALYSIS.csv"
    results_df.to_csv(output_file, index=False)
    print("\n" + "=" * 120)
    print(f"Full results saved to: {output_file}")
    print("=" * 120)


if __name__ == "__main__":
    main()