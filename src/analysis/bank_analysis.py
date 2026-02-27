import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple


class BankFraudAnalyzer:
    
    def __init__(self):
        self.results: Dict[str, pd.DataFrame] = {}
        
    def analyze_kosovo_banks(
        self,
        df: pd.DataFrame,
        score_column: str = 'aml_score',
        anomaly_column: str = 'anomaly_label',
        min_transactions: int = 5
    ) -> pd.DataFrame:
        
        required_cols = {
            'amount',
            score_column,
            anomaly_column,
            'sender_country',
            'receiver_country',
            'sender_bank',
            'receiver_bank',
        }
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            self.results['kosovo_banks_missing_columns'] = pd.DataFrame(
                {'missing_column': missing}
            )
            return pd.DataFrame()

        kosovo_mask = (
            (df['sender_country'].str.upper() == 'XK') |
            (df['receiver_country'].str.upper() == 'XK')
        )
        df_kosovo = df[kosovo_mask].copy()
        
        if len(df_kosovo) == 0:
            return pd.DataFrame()
        
        sender_in_kosovo = df_kosovo[df_kosovo['sender_country'].str.upper() == 'XK']
        receiver_in_kosovo = df_kosovo[df_kosovo['receiver_country'].str.upper() == 'XK']
        
        sender_analysis = pd.DataFrame()
        if len(sender_in_kosovo) > 0:
            sender_analysis = self._analyze_banks_by_role(
                sender_in_kosovo,
                bank_col='sender_bank',
                country_col='sender_country',
                score_column=score_column,
                anomaly_column=anomaly_column,
                role='sender',
                min_transactions=min_transactions
            )
        
        receiver_analysis = pd.DataFrame()
        if len(receiver_in_kosovo) > 0:
            receiver_analysis = self._analyze_banks_by_role(
                receiver_in_kosovo,
                bank_col='receiver_bank',
                country_col='receiver_country',
                score_column=score_column,
                anomaly_column=anomaly_column,
                role='receiver',
                min_transactions=min_transactions
            )
        
        if len(sender_analysis) > 0 and len(receiver_analysis) > 0:
            combined = pd.concat([sender_analysis, receiver_analysis], ignore_index=True)
        elif len(sender_analysis) > 0:
            combined = sender_analysis
        elif len(receiver_analysis) > 0:
            combined = receiver_analysis
        else:
            return pd.DataFrame()
        
        bank_metrics = combined.groupby('bank').agg({
            'total_transactions': 'sum',
            'flagged_transactions': 'sum',
            'avg_aml_score': 'mean',
            'max_aml_score': 'max',
            'total_amount': 'sum'
        }).reset_index()
        
        bank_metrics['flagged_percent'] = (
            bank_metrics['flagged_transactions'] / bank_metrics['total_transactions'] * 100
        )
        
        bank_metrics['risk_score'] = (
            bank_metrics['flagged_percent'] * 0.4 +
            bank_metrics['avg_aml_score'] * 100 * 0.3 +
            bank_metrics['max_aml_score'] * 100 * 0.3
        )

        bank_metrics['has_fraud'] = bank_metrics['flagged_transactions'] > 0
        bank_metrics = bank_metrics.sort_values(
            by=['has_fraud', 'flagged_transactions', 'risk_score'],
            ascending=[False, False, False]
        ).drop(columns=['has_fraud'])
        
        self.results['kosovo_banks'] = bank_metrics
        
        return bank_metrics
    
    def _analyze_banks_by_role(
        self,
        df: pd.DataFrame,
        bank_col: str,
        country_col: str,
        score_column: str,
        anomaly_column: str,
        role: str,
        min_transactions: int
    ) -> pd.DataFrame:
        
        bank_groups = df.groupby(bank_col).agg({
            'amount': ['count', 'sum'],
            score_column: ['mean', 'max'],
            anomaly_column: 'sum'
        })
        
        bank_groups.columns = [
            'total_transactions',
            'total_amount',
            'avg_aml_score',
            'max_aml_score',
            'flagged_transactions'
        ]
        bank_groups = bank_groups.reset_index()
        bank_groups.columns = [
            'bank',
            'total_transactions',
            'total_amount',
            'avg_aml_score',
            'max_aml_score',
            'flagged_transactions'
        ]
        
        bank_groups = bank_groups[bank_groups['total_transactions'] >= min_transactions]
        
        bank_groups['role'] = role
        
        return bank_groups
    
    def analyze_all_banks(
        self,
        df: pd.DataFrame,
        score_column: str = 'aml_score',
        anomaly_column: str = 'anomaly_label',
        min_transactions: int = 10,
        top_n: int = 20
    ) -> pd.DataFrame:
        
        required_cols = {
            'amount',
            score_column,
            anomaly_column,
            'sender_bank',
            'receiver_bank',
        }
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            self.results['all_banks_missing_columns'] = pd.DataFrame(
                {'missing_column': missing}
            )
            return pd.DataFrame()
       
        sender_analysis = self._analyze_banks_by_role(
            df,
            bank_col='sender_bank',
            country_col='sender_country',
            score_column=score_column,
            anomaly_column=anomaly_column,
            role='sender',
            min_transactions=min_transactions
        )
        
        receiver_analysis = self._analyze_banks_by_role(
            df,
            bank_col='receiver_bank',
            country_col='receiver_country',
            score_column=score_column,
            anomaly_column=anomaly_column,
            role='receiver',
            min_transactions=min_transactions
        )
        
        if len(sender_analysis) == 0 and len(receiver_analysis) == 0:
            return pd.DataFrame()

        combined = pd.concat([sender_analysis, receiver_analysis], ignore_index=True)
        
        bank_metrics = combined.groupby('bank').agg({
            'total_transactions': 'sum',
            'flagged_transactions': 'sum',
            'avg_aml_score': 'mean',
            'max_aml_score': 'max',
            'total_amount': 'sum'
        }).reset_index()
        
        bank_metrics['flagged_percent'] = (
            bank_metrics['flagged_transactions'] / bank_metrics['total_transactions'] * 100
        )
        
        bank_metrics['risk_score'] = (
            bank_metrics['flagged_percent'] * 0.4 +
            bank_metrics['avg_aml_score'] * 100 * 0.3 +
            bank_metrics['max_aml_score'] * 100 * 0.3
        )

        # Same sorting rule: fraud banks first
        bank_metrics['has_fraud'] = bank_metrics['flagged_transactions'] > 0
        bank_metrics = bank_metrics.sort_values(
            by=['has_fraud', 'flagged_transactions', 'risk_score'],
            ascending=[False, False, False]
        ).drop(columns=['has_fraud']).head(top_n)
        
        self.results['all_banks'] = bank_metrics
        
        return bank_metrics
    
    def create_bank_chart(
        self,
        bank_metrics: pd.DataFrame,
        metric: str = 'risk_score',
        top_n: int = 10,
        title: str = "Top Fraudulent Banks"
    ) -> go.Figure:
        
        df_plot = bank_metrics.head(top_n).copy()
        
        metric_labels = {
            'risk_score': 'Risk Score',
            'flagged_percent': 'Flagged %',
            'avg_aml_score': 'Avg AML Score',
            'total_transactions': 'Total Transactions',
            'flagged_transactions': 'Flagged Transactions'
        }
        
        y_label = metric_labels.get(metric, metric)
        
        fig = px.bar(
            df_plot,
            x='bank',
            y=metric,
            color=metric,
            color_continuous_scale='Reds',
            title=title,
            labels={'bank': 'Bank', metric: y_label},
            hover_data={
                'total_transactions': ':,',
                'flagged_transactions': ':,',
                'flagged_percent': ':.2f',
                'avg_aml_score': ':.3f',
                'risk_score': ':.2f'
            }
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500,
            showlegend=False,
            hovermode='x'
        )
        
        return fig
    
    def create_comparison_chart(
        self,
        bank_metrics: pd.DataFrame,
        top_n: int = 10
    ) -> go.Figure:
        
        df_plot = bank_metrics.head(top_n).copy()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Flagged %',
            x=df_plot['bank'],
            y=df_plot['flagged_percent'],
            marker_color='indianred'
        ))
        
        fig.add_trace(go.Bar(
            name='Avg AML Score (×100)',
            x=df_plot['bank'],
            y=df_plot['avg_aml_score'] * 100,
            marker_color='lightsalmon'
        ))
        
        fig.update_layout(
            title='Bank Fraud Metrics Comparison',
            xaxis_tickangle=-45,
            barmode='group',
            height=500,
            yaxis_title='Value',
            xaxis_title='Bank',
            hovermode='x unified'
        )
        
        return fig


def analyze_kosovo_banks(
    df: pd.DataFrame,
    score_column: str = 'aml_score',
    anomaly_column: str = 'anomaly_label',
    min_transactions: int = 5
) -> Tuple[pd.DataFrame, go.Figure]:
    analyzer = BankFraudAnalyzer()

    try:
        metrics = analyzer.analyze_kosovo_banks(
            df,
            score_column=score_column,
            anomaly_column=anomaly_column,
            min_transactions=min_transactions
        )
    except KeyError as e:
        print(f"[Bank analysis] Missing column for Kosovo analysis: {e}")
        metrics = pd.DataFrame()
    except Exception as e:
        print(f"[Bank analysis] Unexpected error in analyze_kosovo_banks: {e}")
        metrics = pd.DataFrame()

    if not metrics.empty:
        fig = analyzer.create_bank_chart(
            metrics,
            metric='risk_score',
            top_n=10,
            title="Top 10 Most Fraudulent Banks in Kosovo"
        )
    else:
        fig = go.Figure()
        fig.update_layout(
            title="No Kosovo bank data available",
            xaxis_title="",
            yaxis_title=""
        )

    return metrics, fig


if __name__ == "__main__":
    print("Bank analysis module ready")

