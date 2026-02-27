import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, cast
from difflib import get_close_matches
import json

FEATURE_SCHEMA = {
    "transaction_id": ["transaction_id", "tx_id", "id"],
    "amount": ["amount", "transaction_value", "amt", "shuma_e_transaksionit", "shuma"],
    "sender_id": ["sender_id", "from_id", "id_i_llogarise_se_derguesit", "sender_account"],
    "receiver_id": ["receiver_id", "to_id", "id_i_llogarise_se_perfituesit", "receiver_account"],
    "sender_bank": ["sender_bank", "bank_from", "banka_derguese", "sending_bank"],
    "receiver_bank": ["receiver_bank", "bank_to", "banka_pranuese", "receiving_bank"],
    "sender_country": ["sender_country", "origin_country", "lokacioni_i_derguesit", "from_country"],
    "receiver_country": ["receiver_country", "dest_country", "lokacioni_i_perfituesit", "to_country"],
    "timestamp": ["timestamp", "date", "transaction_date", "data_transaksionit", "datetime"],
    "transaction_type": ["transaction_type", "type", "tx_type"],
    "currency": ["currency", "valuta_e_transaksionit", "valuta"],
    "sender_details": ["sender_details", "detajet_e_tjera_te_derguesit"],
    "receiver_details": ["receiver_details", "detajet_e_tjera_te_perfituesit"],
}

REQUIRED_FEATURES = ["amount", "timestamp"]


class DataLoader:
    
    def __init__(self, schema: Optional[Dict] = None):
        
        self.schema = schema or FEATURE_SCHEMA
        self.mapping = {}
        
    def load_data(self, file_path) -> pd.DataFrame:
        if hasattr(file_path, 'name'):
            filename = file_path.name
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {filename}")
        else:
            # It's a file path string
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
        return df
    
    def normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy.columns = [str(c).strip().lower().replace(' ', '_') for c in df_copy.columns]
        return df_copy
    
    def auto_map_features(self, df: pd.DataFrame, cutoff: float = 0.6) -> Dict[str, Optional[str]]:
        
        df_normalized = self.normalize_column_names(df)
        available_cols = df_normalized.columns.tolist()
        
        mapping = {}
        
        for canonical_name, variants in self.schema.items():
            normalized_variants = [v.lower().replace(' ', '_') for v in variants]
            
            matched = False
            for variant in normalized_variants:
                if variant in available_cols:
                    mapping[canonical_name] = variant
                    matched = True
                    break
            
            if not matched:
                matches = get_close_matches(canonical_name, available_cols, n=1, cutoff=cutoff)
                if matches:
                    mapping[canonical_name] = matches[0]
                else:
                    # Try fuzzy match against variants
                    for variant in normalized_variants:
                        matches = get_close_matches(variant, available_cols, n=1, cutoff=cutoff)
                        if matches:
                            mapping[canonical_name] = matches[0]
                            matched = True
                            break
                    
                    if not matched:
                        mapping[canonical_name] = None
        
        self.mapping = mapping
        return mapping
    
    def apply_mapping(self, df: pd.DataFrame, mapping: Optional[Dict[str, Optional[str]]] = None) -> pd.DataFrame:
       
        df_normalized = self.normalize_column_names(df)
        use_mapping = mapping if mapping is not None else self.mapping
        
        normalized_mapping = {}
        for canonical, raw_col in use_mapping.items():
            if raw_col is not None:
                normalized_col = str(raw_col).lower().replace(' ', '_')
                normalized_mapping[canonical] = normalized_col
            else:
                normalized_mapping[canonical] = None
        
        rename_dict = {v: k for k, v in normalized_mapping.items() if v is not None}
        
        df_mapped = df_normalized.rename(columns=rename_dict)
        
        mapped_canonical_names = [k for k, v in normalized_mapping.items() if v is not None]
        
        available_mapped_cols = [col for col in mapped_canonical_names if col in df_mapped.columns]
        
        result_df: pd.DataFrame
        if len(available_mapped_cols) == 0:
            result_df = pd.DataFrame(index=df_mapped.index)
        elif len(available_mapped_cols) == 1:
            single_col_df = df_mapped[[available_mapped_cols[0]]]
            result_df = cast(pd.DataFrame, single_col_df.copy())
        else:
            multi_col_df = df_mapped.loc[:, available_mapped_cols]
            result_df = cast(pd.DataFrame, multi_col_df.copy())
        
        return result_df
    
    def update_manual_mapping(self, manual_mapping: Dict[str, Optional[str]], df: Optional[pd.DataFrame] = None) -> Dict[str, Optional[str]]:
       
        validated_mapping = {}
        
        valid_cols = set()
        if df is not None:
            df_normalized = self.normalize_column_names(df)
            valid_cols = set(df_normalized.columns)
        
        for canonical_name in self.schema.keys():
            selected_col = manual_mapping.get(canonical_name)
            
            if selected_col and selected_col != '(Not Mapped)':
                normalized_col = str(selected_col).lower().replace(' ', '_')
                
                if df is not None and normalized_col not in valid_cols:
                    validated_mapping[canonical_name] = None
                else:
                    validated_mapping[canonical_name] = normalized_col
            else:
                validated_mapping[canonical_name] = None
        
        self.mapping = validated_mapping
        
        return validated_mapping
    
    def validate_required_mapping(self, mapping: Optional[Dict[str, Optional[str]]] = None, 
                                   required_features: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
    
        use_mapping = mapping if mapping is not None else self.mapping
        use_required = required_features if required_features is not None else REQUIRED_FEATURES
        
        missing = [feat for feat in use_required if not use_mapping.get(feat)]
        
        return len(missing) == 0, missing
    
    def get_mapping_confidence(self, df: pd.DataFrame) -> Dict[str, float]:
        
        df_normalized = self.normalize_column_names(df)
        available_cols = set(df_normalized.columns.tolist())
        
        confidence = {}
        
        for canonical_name, variants in self.schema.items():
            normalized_variants = [v.lower().replace(' ', '_') for v in variants]
            
            mapped_col = self.mapping.get(canonical_name)
            
            if mapped_col is None:
                confidence[canonical_name] = 0.0
            elif mapped_col not in available_cols:
                confidence[canonical_name] = 0.0
            elif any(mapped_col == v for v in normalized_variants):
                confidence[canonical_name] = 1.0
            else:
                from difflib import SequenceMatcher
                max_ratio = max([
                    SequenceMatcher(None, mapped_col, v).ratio()
                    for v in normalized_variants
                ])
                confidence[canonical_name] = max_ratio
        
        return confidence
    
    def validate_schema(self, df: pd.DataFrame, required_features: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
     
        if required_features is None:
            required_features = ['amount', 'timestamp']
        
        missing = [f for f in required_features if f not in df.columns]
        
        return len(missing) == 0, missing
    
    def get_unmapped_columns(self, df: pd.DataFrame) -> List[str]:
       
        df_normalized = self.normalize_column_names(df)
        mapped_cols = set(v for v in self.mapping.values() if v is not None)
        
        return [col for col in df_normalized.columns if col not in mapped_cols]
    
    def save_mapping(self, file_path: str):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.mapping, f, indent=2)
    
    def load_mapping(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.mapping = json.load(f)


def load_and_map_data(file_path: str, custom_mapping: Optional[Dict[str, Optional[str]]] = None) -> Tuple[pd.DataFrame, Dict[str, Optional[str]], Dict[str, float]]:
   
    loader = DataLoader()
    df = loader.load_data(file_path)
    
    if custom_mapping:
        loader.mapping = custom_mapping
        df_mapped = loader.apply_mapping(df, custom_mapping)
    else:
        mapping = loader.auto_map_features(df)
        df_mapped = loader.apply_mapping(df)
    
    confidence = loader.get_mapping_confidence(df)
    
    return df_mapped, loader.mapping, confidence


if __name__ == "__main__":
    loader = DataLoader()
    
    print("DataLoader module ready for import")

