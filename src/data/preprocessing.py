# src/data/preprocessing.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class AMLPreprocessor:

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self.get_default_config()

    @staticmethod
    def get_default_config() -> Dict:
        return {
            'reporting_threshold': 10000.0,
            'near_threshold_buffer': 0.9,  # 90% of threshold
            'structuring_min_count': 3,
            'multi_bank_min': 3,
            'high_amount_percentile': 0.99,
            'contamination': 0.01,
            'midnight_start_hour': 0,
            'midnight_end_hour': 6,
            'high_risk_countries': [
                "KP", "IR", "MM", "DZ", "AO", "BG", "BF", "CM", "CI", "CD",
                "HT", "KE", "LA", "LB", "ML", "MC", "MZ", "NA", "NP", "NG",
                "ZA", "SS", "SY", "TZ", "VE", "VN", "YE", "BO", "VI"
            ],
            'high_risk_drug_countries': [
                'AF', 'CO', 'MX', 'VE', 'MM', 'BO', 'PE', 'LA', 'PA'
            ],
            'weight_high_risk_sender': 3.0,
            'weight_high_risk_receiver': 3.0,
            'weight_drug_country': 4.0,
            'weight_near_threshold': 2.5,
            'weight_structuring': 2.0,
            'weight_midnight': 1.5,
            'weight_high_cashflow': 2.5,
            'weight_multiple_high_risk': 2.0,
            'weight_rare_bank': 1.5,
        }

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()

        if 'amount' in df_clean.columns:
            df_clean['amount'] = pd.to_numeric(df_clean['amount'], errors='coerce')
            df_clean = df_clean[df_clean['amount'] > 0]

        if 'timestamp' in df_clean.columns:
            df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'], errors='coerce')

        df_clean = df_clean.drop_duplicates(keep='first')

        if 'sender_id' in df_clean.columns and 'receiver_id' in df_clean.columns:
            df_clean = df_clean.dropna(subset=['sender_id', 'receiver_id'])

        if 'sender_country' in df_clean.columns:
            df_clean['sender_country'] = df_clean['sender_country'].astype(str).str.upper().str.strip()
        if 'receiver_country' in df_clean.columns:
            df_clean['receiver_country'] = df_clean['receiver_country'].astype(str).str.upper().str.strip()

        df_clean = df_clean.fillna({
            'sender_bank': '',
            'receiver_bank': '',
            'sender_country': '',
            'receiver_country': '',
            'transaction_type': 'Unknown',
            'currency': 'EUR'
        })

        return df_clean

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_eng = df.copy()

        if 'timestamp' in df_eng.columns:
            ts = pd.to_datetime(df_eng['timestamp'], errors='coerce')
            df_eng['timestamp'] = ts
            df_eng['hour'] = ts.dt.hour.fillna(12).astype(int)
            df_eng['day'] = ts.dt.day
            df_eng['month'] = ts.dt.month
            df_eng['year'] = ts.dt.year
            df_eng['dayofweek'] = ts.dt.dayofweek
            df_eng['date'] = ts.dt.floor('D')
        else:
            df_eng['hour'] = 12
            df_eng['day'] = 1
            df_eng['month'] = 1
            df_eng['year'] = 2025
            df_eng['dayofweek'] = 0
            df_eng['date'] = pd.NaT

        high_risk = [c.upper() for c in self.config['high_risk_countries']]
        high_risk_drug = [c.upper() for c in self.config['high_risk_drug_countries']]

        # --- High-risk flags: handle missing country columns gracefully ---
        if 'sender_country' in df_eng.columns:
            df_eng['high_risk_sender'] = df_eng['sender_country'].isin(high_risk).astype(int)
        else:
            # Column not mapped → default to 0
            df_eng['high_risk_sender'] = 0

        if 'receiver_country' in df_eng.columns:
            df_eng['high_risk_receiver'] = df_eng['receiver_country'].isin(high_risk).astype(int)
            df_eng['is_high_risk_drug_country_receiver'] = df_eng['receiver_country'].isin(high_risk_drug).astype(int)
        else:
            df_eng['high_risk_receiver'] = 0
            df_eng['is_high_risk_drug_country_receiver'] = 0

        reporting_threshold = float(self.config['reporting_threshold'])
        buffer = float(self.config['near_threshold_buffer'])

        df_eng['near_threshold'] = (
            (df_eng['amount'] >= reporting_threshold * buffer) &
            (df_eng['amount'] <= reporting_threshold)
        ).astype(int)

        df_eng['small_transaction'] = (df_eng['amount'] < reporting_threshold * buffer).astype(int)

        if all(c in df_eng.columns for c in ['sender_id', 'date']):
            df_eng['small_txn_count_per_day'] = (
                df_eng.groupby(['sender_id', 'date'])['small_transaction'].transform('sum')
            ).astype(int)

            df_eng['structuring_flag'] = (
                df_eng['small_txn_count_per_day'] >= int(self.config['structuring_min_count'])
            ).astype(int)
        else:
            df_eng['small_txn_count_per_day'] = 0
            df_eng['structuring_flag'] = 0

        if all(c in df_eng.columns for c in ['sender_id', 'date']):
            df_eng['high_risk_transfers_per_day'] = (
                df_eng.groupby(['sender_id', 'date'])['is_high_risk_drug_country_receiver'].transform('sum')
            ).astype(int)

            df_eng['multiple_high_risk_transfers'] = (df_eng['high_risk_transfers_per_day'] >= 2).astype(int)
        else:
            df_eng['high_risk_transfers_per_day'] = 0
            df_eng['multiple_high_risk_transfers'] = 0

        if all(c in df_eng.columns for c in ['sender_bank', 'receiver_bank', 'sender_country', 'receiver_country']):
            df_eng['sender_bank_count'] = (
                df_eng.groupby(['sender_country', 'sender_bank'])['amount'].transform('size')
            ).fillna(0).astype(int)

            df_eng['receiver_bank_count'] = (
                df_eng.groupby(['receiver_country', 'receiver_bank'])['amount'].transform('size')
            ).fillna(0).astype(int)

            df_eng['rare_sender_bank_combo'] = (
                df_eng['sender_bank_count'] < int(self.config['multi_bank_min'])
            ).astype(int)
            df_eng['rare_receiver_bank_combo'] = (
                df_eng['receiver_bank_count'] < int(self.config['multi_bank_min'])
            ).astype(int)

            df_eng['rare_bank_combo_flag'] = (
                (df_eng['rare_sender_bank_combo'] == 1) |
                (df_eng['rare_receiver_bank_combo'] == 1)
            ).astype(int)
        else:
            df_eng['sender_bank_count'] = 0
            df_eng['receiver_bank_count'] = 0
            df_eng['rare_sender_bank_combo'] = 0
            df_eng['rare_receiver_bank_combo'] = 0
            df_eng['rare_bank_combo_flag'] = 0


        midnight_start = int(self.config.get('midnight_start_hour', 0))
        midnight_end = int(self.config.get('midnight_end_hour', 6))
        df_eng['midnight_transaction'] = (
            (df_eng['hour'] >= midnight_start) & (df_eng['hour'] <= midnight_end)
        ).astype(int)

        threshold_single = float(df_eng['amount'].quantile(self.config['high_amount_percentile']))
        df_eng['high_cashflow_flag_acct_total'] = (df_eng['amount'] >= threshold_single).astype(int)

        df_eng['high_cashflow_midnight'] = (
            (df_eng['high_cashflow_flag_acct_total'] == 1) &
            (df_eng['midnight_transaction'] == 1)
        ).astype(int)

        if all(c in df_eng.columns for c in ['sender_id', 'date', 'receiver_country']):
            df_eng['unique_receiver_locations_per_day'] = (
                df_eng.groupby(['sender_id', 'date'])['receiver_country'].transform('nunique')
            ).fillna(0).astype(int)

            df_eng['transaksione_ne_lokacione_ndryshe'] = (df_eng['unique_receiver_locations_per_day'] > 1).astype(int)
        else:
            df_eng['unique_receiver_locations_per_day'] = 0
            df_eng['transaksione_ne_lokacione_ndryshe'] = 0

        if all(c in df_eng.columns for c in ['receiver_id', 'date', 'sender_id']):
            df_eng['unique_senders_per_day'] = (
                df_eng.groupby(['receiver_id', 'date'])['sender_id'].transform('nunique')
            ).fillna(0).astype(int)

            df_eng['multiple_senders_same_receiver'] = (df_eng['unique_senders_per_day'] > 1).astype(int)
        else:
            df_eng['unique_senders_per_day'] = 0
            df_eng['multiple_senders_same_receiver'] = 0

        w = self.config
        df_eng['enhanced_aml_score'] = (
            df_eng['high_risk_sender'] * w.get('weight_high_risk_sender', 3.0) +
            df_eng['high_risk_receiver'] * w.get('weight_high_risk_receiver', 3.0) +
            df_eng['is_high_risk_drug_country_receiver'] * w.get('weight_drug_country', 4.0) +
            df_eng['high_cashflow_flag_acct_total'] * w.get('weight_high_cashflow', 2.5) +
            df_eng['near_threshold'] * w.get('weight_near_threshold', 2.5) +
            df_eng['structuring_flag'] * w.get('weight_structuring', 2.0) +
            df_eng['midnight_transaction'] * w.get('weight_midnight', 1.5) +
            df_eng['multiple_high_risk_transfers'] * w.get('weight_multiple_high_risk', 2.0) +
            df_eng['rare_bank_combo_flag'] * w.get('weight_rare_bank', 1.5)
        ).astype(float)

        df_eng['definite_fraud_flag'] = (
            ((df_eng['high_risk_sender'] == 1) | (df_eng['high_risk_receiver'] == 1)) &
            (df_eng['near_threshold'] == 1) &
            (df_eng['midnight_transaction'] == 1)
        ).astype(int)

        df_eng['high_risk_country_transaction'] = (
            (df_eng['high_risk_sender'] == 1) | (df_eng['high_risk_receiver'] == 1)
        ).astype(int)

        df_eng['high_risk_drug_country_transaction'] = (
            df_eng['is_high_risk_drug_country_receiver'] == 1
        ).astype(int)

        df_eng['threshold_risk_transaction'] = (df_eng['near_threshold'] == 1).astype(int)

        flag_cols = [
            'high_risk_sender', 'high_risk_receiver', 'is_high_risk_drug_country_receiver',
            'near_threshold', 'structuring_flag', 'midnight_transaction',
            'high_cashflow_flag_acct_total', 'rare_bank_combo_flag', 'multiple_high_risk_transfers',
            'definite_fraud_flag', 'high_risk_country_transaction',
            'high_risk_drug_country_transaction', 'threshold_risk_transaction',
            'transaksione_ne_lokacione_ndryshe', 'multiple_senders_same_receiver'
        ]
        for c in flag_cols:
            if c in df_eng.columns:
                df_eng[c] = (pd.to_numeric(df_eng[c], errors='coerce').fillna(0) > 0).astype(int)

        count_cols = [
            'small_txn_count_per_day', 'high_risk_transfers_per_day',
            'sender_bank_count', 'receiver_bank_count',
            'unique_receiver_locations_per_day', 'unique_senders_per_day'
        ]
        for c in count_cols:
            if c in df_eng.columns:
                df_eng[c] = pd.to_numeric(df_eng[c], errors='coerce').fillna(0).astype(int)

        return df_eng

    def get_numeric_features(self, df: pd.DataFrame) -> List[str]:
        exclude_cols = [
            'transaction_id', 'timestamp', 'date', 'sender_id', 'receiver_id',
            'sender_bank', 'receiver_bank', 'sender_country', 'receiver_country',
            'transaction_type', 'currency', 'sender_details', 'receiver_details'
        ]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [col for col in numeric_cols if col not in exclude_cols]

    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        df_clean = self.clean_data(df)
        df_eng = self.engineer_features(df_clean)
        numeric_features = self.get_numeric_features(df_eng)
        return df_eng, numeric_features

    def update_config(self, new_config: Dict):
        self.config.update(new_config)


def preprocess_data(df: pd.DataFrame, config: Optional[Dict] = None) -> Tuple[pd.DataFrame, List[str]]:
    preprocessor = AMLPreprocessor(config)
    return preprocessor.preprocess(df)


if __name__ == "__main__":
    print("Preprocessing module ready for import")
