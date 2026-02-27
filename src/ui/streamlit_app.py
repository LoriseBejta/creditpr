import os, sys
from pathlib import Path
import json 
# at the top of streamlit_app.py
from translation import t 



_THIS_FILE = Path(__file__).resolve()           
_SRC_DIR   = _THIS_FILE.parents[1]              
_PROJ_DIR  = _SRC_DIR.parent                    

for p in (str(_SRC_DIR), str(_PROJ_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

_MODELS_DIR = _SRC_DIR / "models"
(_MODELS_DIR / "__init__.py").touch(exist_ok=True)


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import joblib

from src.data.loader import DataLoader, FEATURE_SCHEMA, REQUIRED_FEATURES
from src.data.preprocessing import AMLPreprocessor
from src.models.isolation_forest import IsolationForestModel
from src.models.autoencoder import AutoencoderModel
from src.models.train import ModelTrainer
from src.models.retrain import ModelRetrainer
from src.viz.network_graph import TransactionNetworkGraph, create_transaction_network
from src.analysis.bank_analysis import BankFraudAnalyzer, analyze_kosovo_banks



st.set_page_config(
    page_title="AML Fraud Detection Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)



def init_session_state():
    defaults = {
        'language': 'en',
        'data_loader': DataLoader(),
        'preprocessor': AMLPreprocessor(),
        'trainer': ModelTrainer(),
        'retrainer': ModelRetrainer(),
        'uploaded_data': None,
        'processed_data': None,
        'mapping': {},
        'mapping_confidence': {},
        'scored_data': None,
        'selected_model': None,
        'models_dir': 'models',
        'config': AMLPreprocessor.get_default_config(),
        'training_log': [],
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()



with st.sidebar: 
    lang_options = {
        'en': 'EN English',
        'al': 'AL Shqip'
    }
    selected_lang = st.selectbox(
        t('language'),
        options=list(lang_options.keys()),
        format_func=lambda x: lang_options[x],
        index=0 if st.session_state.language == 'en' else 1,
        key='language_selector'
    )
    if selected_lang != st.session_state.language:
        st.session_state.language = selected_lang
        st.rerun()
    
    st.markdown("---")
    
    if st.session_state.scored_data is not None:
        df = st.session_state.scored_data
        st.metric(t('total_transactions'), f"{len(df):,}")
        
        if 'is_flagged' in df.columns:
            n_flagged = df['is_flagged'].sum()
            flagged_rate = (n_flagged/len(df)*100) if len(df) > 0 else 0
            st.metric(t('flagged_transactions'), f"{n_flagged:,}", delta=f"{flagged_rate:.2f}%")
        elif 'anomaly_label' in df.columns:
            n_flagged = df['anomaly_label'].sum()
            flagged_rate = (n_flagged/len(df)*100) if len(df) > 0 else 0
            st.metric(t('flagged_transactions'), f"{n_flagged:,}", delta=f"{flagged_rate:.2f}%")
    st.markdown("---")
    st.caption(t('sidebar_footer'))




st.title(t('app_title'))
st.markdown(f"**{t('app_subtitle')}**")
st.markdown("---")


tabs = st.tabs([
    t('tab_indicators'),
    t('tab_overview'),
    t('tab_models'),
    t('tab_network'),
    t('tab_analysis'),
    t('tab_logs')
])

tab_indicators, tab_overview, tab_models, tab_network, tab_analysis, tab_logs = tabs



with tab_overview:
    st.header(t('tab_overview'))
    
    uploaded_file = st.file_uploader(
        t('upload_file'),
        type=['csv', 'xlsx', 'xls'],
        help=t('file_info')
    )
    
    if uploaded_file is not None:
        with st.spinner(t('processing')):
            try:
                df = st.session_state.data_loader.load_data(uploaded_file)
                st.session_state.uploaded_data = df
                
                file_hash = str(hash((uploaded_file.name, len(df), tuple(df.columns))))
                is_new_file = st.session_state.get('file_hash') != file_hash
                
                if is_new_file:
                    st.session_state.file_hash = file_hash
                    st.session_state.mapping_mode = "auto"
                    mapping = st.session_state.data_loader.auto_map_features(df)
                    confidence = st.session_state.data_loader.get_mapping_confidence(df)
                    st.session_state.mapping = mapping
                    st.session_state.mapping_confidence = confidence
                elif st.session_state.get('mapping_mode') != "manual":
                    mapping = st.session_state.data_loader.auto_map_features(df)
                    confidence = st.session_state.data_loader.get_mapping_confidence(df)
                    st.session_state.mapping = mapping
                    st.session_state.mapping_confidence = confidence
                    st.session_state.mapping_mode = "auto"
                
                st.success(t("load_success", n=len(df)))
                
            except Exception as e:
                st.error(f"{t('error')}: {str(e)}")
                df = None
                mapping = {}
                confidence = {}
        
        if df is not None:
            st.subheader(t('mapping_confidence'))
            mapping = st.session_state.mapping
            confidence = st.session_state.mapping_confidence
            
            conf_data = []
            for k, v in mapping.items():
                mapped_col = v if v is not None else 'Not Mapped'
                conf_score = f"{confidence.get(k, 0)*100:.0f}%"
                is_required = "⭐" if k in REQUIRED_FEATURES else ""
                conf_data.append({
                    'Required': is_required,
                    'Feature': k, 
                    'Mapped Column': mapped_col, 
                    'Confidence': conf_score
                })
            conf_df = pd.DataFrame(conf_data)
            
            def color_confidence(val):
                if 'Not Mapped' in str(val):
                    return 'background-color: #ffcccc'
                elif '100%' in str(val):
                    return 'background-color: #ccffcc'
                elif '%' in str(val):
                    try:
                        pct_val = int(str(val).strip('%'))
                        if pct_val >= 70:
                            return 'background-color: #ffffcc'
                        else:
                            return 'background-color: #ffddcc'
                    except:
                        return ''
                return ''
            
            st.dataframe(
                conf_df.style.map(color_confidence, subset=['Confidence', 'Mapped Column']),
                use_container_width=True,
                hide_index=True
            )
            
            with st.expander(t('adjust_mappings_expander')):
                df_normalized = st.session_state.data_loader.normalize_column_names(df)
                cols = df_normalized.columns.tolist()
                file_ns = st.session_state.get("file_hash", str(hash(tuple(cols))))
                current_mapping = st.session_state.mapping
                mapped_count = sum(1 for v in current_mapping.values() if v is not None)
                required_count = sum(1 for feat in REQUIRED_FEATURES if current_mapping.get(feat))
                
                st.write(t('currently_mapped').format(mapped_count=mapped_count,total_features=len(FEATURE_SCHEMA)))
                st.info(t('required_fields_info').format(required=', '.join(REQUIRED_FEATURES),mapped=required_count,total_required=len(REQUIRED_FEATURES)))
                st.info(t('auto_mapping_tip'))
                
                for canonical in FEATURE_SCHEMA.keys():
                    current = current_mapping.get(canonical, None)
                    options = ['(Not Mapped)'] + cols
                    idx = cols.index(current) + 1 if current and current in cols else 0
                    label = f"{canonical}:" if canonical not in REQUIRED_FEATURES else f"{canonical}: ⭐ REQUIRED"
                    st.selectbox(
                        label,
                        options=options,
                        index=idx,
                        key=f"map_{file_ns}_{canonical}"
                    )
                
                if st.button("✅ Apply Manual Mapping", type="primary"):
                    new_mapping = {}
                    mapped_features = []
                    for canonical in FEATURE_SCHEMA.keys():
                        key_name = f"map_{file_ns}_{canonical}"
                        selected = st.session_state.get(key_name, '(Not Mapped)')
                        if selected and selected != '(Not Mapped)':
                            new_mapping[canonical] = selected
                            mapped_features.append(canonical)
                        else:
                            new_mapping[canonical] = None
                    try:
                        validated_mapping = st.session_state.data_loader.update_manual_mapping(new_mapping, df)
                        st.session_state.mapping = validated_mapping.copy()
                        st.session_state.mapping_confidence = st.session_state.data_loader.get_mapping_confidence(df)
                        st.session_state.mapping_mode = "manual"
                        required_mapped = sum(1 for feat in REQUIRED_FEATURES if validated_mapping.get(feat))
                        st.success(f"✅ **Manual mapping applied successfully!**")
                        st.info(f"**Mapped {len(mapped_features)} out of {len(FEATURE_SCHEMA)} features** | Required fields: {required_mapped}/{len(REQUIRED_FEATURES)}")
                        if mapped_features:
                            with st.expander("📋 View mapped features", expanded=True):
                                for feat in mapped_features:
                                    is_required = "⭐" if feat in REQUIRED_FEATURES else "  "
                                    st.write(f"{is_required} **{feat}** → `{validated_mapping[feat]}`")
                        unmapped_features = [k for k, v in validated_mapping.items() if v is None]
                        if unmapped_features:
                            st.info(f"ℹ️ **{len(unmapped_features)} features not mapped:** {', '.join(unmapped_features)}")
                        if required_mapped < len(REQUIRED_FEATURES):
                            missing = [f for f in REQUIRED_FEATURES if not validated_mapping.get(f)]
                            st.warning(f"⚠️ Missing required fields: **{', '.join(missing)}**. You must map these before processing data.")
                        else:
                            st.success("✅ All required fields are mapped! You can now process the data.")
                    except Exception as e:
                        st.error(f"Mapping error: {str(e)}")
            
            st.markdown("---")
            if st.button("🔄 Process Data", type="primary"):
                is_valid, missing_required = st.session_state.data_loader.validate_required_mapping(st.session_state.mapping)
                if not is_valid:
                    st.error(f"⚠️ **Cannot process data: Missing required fields**")
                    st.warning(f"The following required fields must be mapped: **{', '.join(missing_required)}**")
                    st.info("💡 Please use the manual mapping section above to map these required fields.")
                else:
                    with st.spinner("Processing data..."):
                        try:
                            current_mapping = st.session_state.mapping
                            df_mapped = st.session_state.data_loader.apply_mapping(df, current_mapping)
                            preprocessor = AMLPreprocessor(st.session_state.config)
                            df_processed, numeric_features = preprocessor.preprocess(df_mapped)
                            st.session_state.processed_data = df_processed
                            st.session_state.numeric_features = numeric_features
                            st.success(f"✅ **Mapped {len(df_mapped.columns)} columns from your data:** {', '.join(df_mapped.columns.tolist())}")
                            st.success(f"✅ **Data processing complete!**")
                        except Exception as e:
                            st.error(f"Processing error: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
            
            if st.session_state.processed_data is not None:
                st.subheader(t('data_preview'))
                st.dataframe(st.session_state.processed_data.head(100), use_container_width=True)



with tab_indicators:
    st.header(t('tab_indicators'))
    st.markdown(t("indicators_config_warning"))
    st.markdown("### " + t('indicators_config'))
    config = st.session_state.config
    
    st.markdown(f"#### 🎯 {t('core_thresholds')}")
    col1, col2 = st.columns(2)
    
    with col1:
        config['reporting_threshold'] = st.number_input(
            t('reporting_threshold'),
            min_value=0.0,
            max_value=1000000.0,
            value=float(config['reporting_threshold']),
            step=100.0,
            help="Cash reporting threshold (e.g., €10,000 for Kosovo CBK)"
        )
        config['near_threshold_buffer'] = st.slider(
            t('near_threshold_buffer'),
            min_value=0.5,
            max_value=1.0,
            value=float(config['near_threshold_buffer']),
            step=0.01,
            help="Transactions within this % of threshold are flagged (0.9 = 90-100%)"
        )
    
    with col2:
        config['high_amount_percentile'] = st.slider(
            t('high_amount_percentile'),
            min_value=0.8,
            max_value=0.999,
            value=float(config['high_amount_percentile']),
            step=0.01,
            help="Percentile for high cashflow detection (0.99 = top 1%)" if st.session_state.language == 'en' else "Përqindja për zbulimin e rrjedhjes së lartë të cash (0.99 = top 1%)"
        )
        config['contamination'] = st.slider(
            t('contamination'),
            min_value=0.001,
            max_value=0.2,
            value=float(config['contamination']),
            step=0.001,
            format="%.3f",
            help="Expected fraud rate (0.01 = 1% of transactions flagged)"
        )
    
    st.markdown("---")
    st.markdown(f"#### 🔍 {t('pattern_detection')}")
    col1, col2 = st.columns(2)
    
    with col1:
        config['structuring_min_count'] = st.number_input(
            t('structuring_min'),
            min_value=1,
            max_value=20,
            value=int(config['structuring_min_count']),
            step=1,
            help="Minimum small transactions per day to trigger structuring flag"
        )
        config['multi_bank_min'] = st.number_input(
            t('multi_bank_min'),
            min_value=1,
            max_value=10,
            value=int(config['multi_bank_min']),
            step=1,
            help="Minimum banks per day to trigger multi-bank flag"
        )
    
    with col2:
        midnight_start = st.slider(
            t('midnight_start_hour'),
            min_value=0,
            max_value=23,
            value=0,
            step=1,
            help=t('midnight_start_help')
        )
        midnight_end = st.slider(
            t('midnight_end_hour'),
            min_value=0,
            max_value=23,
            value=6,
            step=1,
            help=t('midnight_end_help')
        )
        config['midnight_start_hour'] = midnight_start
        config['midnight_end_hour'] = midnight_end
    
    st.markdown("---")
    st.markdown(f"#### 🌍 {t('risk_country_lists')}")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**{t('high_risk_countries')}**")
        high_risk_text = st.text_area(
            "Enter country codes (comma-separated)",
            value=", ".join(config.get('high_risk_countries', [])),
            help="Default: KP, IR, MM, DZ, AO, BG, BF, CM, CI, CD, HT, KE, LA, LB, ML, MC, MZ, NA, NP, NG, ZA, SS, SY, TZ, VE, VN, YE, BO, VI",
            height=100
        )
        config['high_risk_countries'] = [c.strip().upper() for c in high_risk_text.split(',') if c.strip()]
    
    with col2:
        st.markdown(f"**{t('drug_trafficking_countries')}**")
        drug_countries_text = st.text_area(
            "Enter country codes (comma-separated)",
            value=", ".join(config.get('high_risk_drug_countries', [])),
            help="Default: AF, CO, MX, VE, MM, BO, PE, LA, PA",
            height=100
        )
        config['high_risk_drug_countries'] = [c.strip().upper() for c in drug_countries_text.split(',') if c.strip()]
    
    st.markdown("---")
    st.markdown(f"#### 💰 {t('feature_weights')}")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        config['weight_high_risk_sender'] = st.number_input(
            t('high_risk_sender_weight'),
            min_value=0.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
            help=t('high_risk_sender_help')
        )
        config['weight_high_risk_receiver'] = st.number_input(
            t('high_risk_receiver_weight'),
            min_value=0.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
            help=t('high_risk_receiver_help')
        )
    
    with col2:
        config['weight_drug_country'] = st.number_input(
            t('drug_country_weight'),
            min_value=0.0,
            max_value=10.0,
            value=4.0,
            step=0.5,
            help=t('drug_country_help')
        )
        config['weight_near_threshold'] = st.number_input(
            t('near_threshold_weight'),
            min_value=0.0,
            max_value=10.0,
            value=2.5,
            step=0.5
        )
    
    with col3:
        config['weight_structuring'] = st.number_input(
            t('structuring_weight'),
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.5
        )
        config['weight_midnight'] = st.number_input(
            t('midnight_weight'),
            min_value=0.0,
            max_value=10.0,
            value=1.5,
            step=0.5
        )
    
    col4, col5 = st.columns(2)
    with col4:
        config['weight_high_cashflow'] = st.number_input(
            t('high_cashflow_weight'),
            min_value=0.0,
            max_value=10.0,
            value=2.5,
            step=0.5,
            help=t('high_cashflow_help')
        )
        config['weight_multiple_high_risk'] = st.number_input(
            t('multiple_high_risk_weight'),
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.5
        )
    with col5:
        config['weight_rare_bank'] = st.number_input(
            t('rare_bank_combo_weight'),
            min_value=0.0,
            max_value=10.0,
            value=1.5,
            step=0.5,
            help=t('rare_bank_help')
        )
    
    if st.button(t('save_config'), type="primary"):
        st.session_state.config = config
        st.session_state.preprocessor.update_config(config)
        st.success(t('config_saved'))
        st.info("💡 **Remember:** Process your data in Overview tab after setting indicators!")
    
    if st.session_state.scored_data is not None:
        st.markdown("---")
        st.markdown("### Live KPIs")
        df = st.session_state.scored_data
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(t('total_transactions'), f"{len(df):,}")
        with col2:
            if 'anomaly_label' in df.columns:
                n_flagged = df['anomaly_label'].sum()
                st.metric(t('flagged_transactions'), f"{n_flagged:,}")
        with col3:
            if 'anomaly_label' in df.columns:
                n_flagged = df['anomaly_label'].sum()
                rate = (n_flagged / len(df)) * 100
                st.metric(t('flagged_rate'), f"{rate:.2f}%")
        with col4:
            if 'aml_score' in df.columns:
                avg = df['aml_score'].mean()
                st.metric(t('avg_score'), f"{avg:.3f}")
        col1, col2 = st.columns(2)
        with col1:
            if 'aml_score' in df.columns:
                fig = px.histogram(df, x='aml_score', nbins=50, title="AML Score Distribution")
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if 'amount' in df.columns:
                fig = px.histogram(df, x='amount', nbins=50, title="Amount Distribution", log_y=True)
                st.plotly_chart(fig, use_container_width=True)



with tab_models:
    import os
    from datetime import datetime
    import numpy as np
    import pandas as pd

    from models.isolation_forest import IsolationForestModel
    from models.autoencoder import AutoencoderModel

    def ui_percentile(scores: np.ndarray) -> np.ndarray:
        if scores is None or len(scores) <= 1:
            return np.zeros_like(scores)
        order = np.argsort(scores)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(scores))
        return ranks / (len(scores) - 1)

    st.header(t('tab_models'))
    os.makedirs(st.session_state.models_dir, exist_ok=True)
    st.subheader(t('train_new_model'))
    
    col1, col2 = st.columns([2, 1])
    with col1:
        model_types = st.multiselect(
            t('select_algorithms'),
            options=['isolation_forest', 'autoencoder'],
            default=['isolation_forest']
        )
    with col2:
        contamination_train = st.number_input(
            "Contamination",
            min_value=0.001,
            max_value=0.5,
            value=float(st.session_state.config['contamination']),
            step=0.001,
            format="%.3f"
        )
    
    if st.button(t('train_button'), type="primary"):
        if st.session_state.processed_data is None:
            st.warning(t('no_data'))
        elif not model_types:
            st.warning("Please select at least one algorithm")
        else:
            with st.spinner(t('training')):
                try:
                    saved_models = st.session_state.trainer.train_from_dataframe(
                        st.session_state.processed_data,
                        model_types,
                        st.session_state.models_dir,
                        float(contamination_train)
                    )
                    log_entry = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'action': 'train',
                        'models': list(saved_models.keys()),
                        'n_samples': int(len(st.session_state.processed_data))
                    }
                    st.session_state.training_log.append(log_entry)
                    st.success(f"✅ Trained {len(saved_models)} model(s)")
                    for model_type, path in saved_models.items():
                        st.write(f"- {model_type}: `{path}`")
                except Exception as e:
                    st.error(f"Training error: {str(e)}")
    
    st.markdown("---")
    st.subheader("Score Transactions")
    
    model_files = []
    models_dir = st.session_state.models_dir or 'models'
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    if model_files:
        selected_model_file = st.selectbox(
            t('model_selection'),
            options=model_files,
            index=0
        )
        
        model_path = os.path.join(models_dir, selected_model_file)

        native_threshold_display = None
        try:
            payload = joblib.load(model_path) 
            native_threshold_display = float(payload.get("threshold", None))
        except Exception:
            pass

        threshold_mode = st.selectbox(
            "Alert policy",
            ["Native (from training)", "Manual cutoff", "Top-k% (capacity)"],
            index=0,
            key="threshold_mode",
        )

        if threshold_mode == "Native (from training)":
            if native_threshold_display is not None:
                st.caption(f"Native threshold (from training): **{native_threshold_display:.6g}**")
            else:
                st.caption("Native threshold will be computed from the training contamination.")

        default_manual = float(native_threshold_display) if native_threshold_display is not None else 0.0
        manual_cutoff = st.number_input(
            "Manual cutoff (model-native units)",
            min_value=0.0, max_value=1e12, value=default_manual, step=0.001,
            help="AE: reconstruction error; IF: inverted score (= -score_samples).",
            disabled=(threshold_mode != "Manual cutoff"),   
            key="manual_cutoff",
        )

        topk_pct = st.slider(
            "Target alert rate (% of txns)",
            min_value=0.10, max_value=5.00, value=1.00, step=0.10,
            disabled=(threshold_mode != "Top-k% (capacity)"),  
            key="topk_pct",
        )
        
        if st.button(t('score_button'), type="primary"):
            if st.session_state.processed_data is None:
                st.warning(t('no_data'))
            else:
                with st.spinner("Scoring transactions..."):
                    try:
                        model_path = os.path.join(models_dir, selected_model_file)
                        if 'autoencoder' in selected_model_file.lower():
                            model = AutoencoderModel.load(model_path)
                        elif 'isolation_forest' in selected_model_file.lower():
                            model = IsolationForestModel.load(model_path)
                        else:
                            try:
                                model = AutoencoderModel.load(model_path)
                            except Exception:
                                model = IsolationForestModel.load(model_path)

                        df = st.session_state.processed_data.copy()
                        features = getattr(model, "features", None)
                        if not features:
                            features = df.select_dtypes(include=[np.number]).columns.tolist()
                        for c in features:
                            if c not in df.columns:
                                df[c] = 0.0
                        X = df[features].fillna(0.0)
                        if not isinstance(X, pd.DataFrame):
                            X = pd.DataFrame(X, columns=features)
                        
                        eps = 1e-9
                        if isinstance(model, AutoencoderModel):
                            native_score = model.reconstruction_error(X)
                            native_threshold = getattr(model, "threshold", None)
                            if native_threshold is None:
                                q = 1.0 - float(st.session_state.config.get("contamination", 0.01))
                                native_threshold = float(np.quantile(native_score, q))
                            ui_score = ui_percentile(native_score)
                        else:
                            native_score = model.native_score(X)  
                            native_threshold = getattr(model, "threshold", None)
                            if native_threshold is None:
                                q = 1.0 - float(st.session_state.config.get("contamination", 0.01))
                                native_threshold = float(np.quantile(native_score, q))
                            ui_score = ui_percentile(native_score) 

                        
                        if threshold_mode == "Native (from training)":
                            model_flag = (native_score >= float(native_threshold))
                            policy_description = f"Native threshold = {float(native_threshold):.6g}"
                        elif threshold_mode == "Manual cutoff":
                            mc = 0.0 if manual_cutoff is None else float(manual_cutoff)
                            model_flag = (native_score >= mc)
                            policy_description = f"Manual cutoff = {mc:.6g}"
                        else:  # Top-k%
                            k = float(topk_pct) / 100.0
                            q = 1.0 - k
                            cut = float(np.quantile(native_score, q))
                            model_flag = (native_score >= cut)
                            policy_description = f"Top-k% policy = top {topk_pct:.1f}% (cut={cut:.6g})"
                        
                        enhanced_threshold = 6.0
                        flagged_by_enhanced = (
                            (df['enhanced_aml_score'] >= enhanced_threshold).astype(int)
                            if 'enhanced_aml_score' in df.columns else
                            np.zeros(len(df), dtype=int)
                        )
                        high_weight_indicators = [
                            'high_risk_sender',
                            'high_risk_receiver',
                            'is_high_risk_drug_country_receiver',
                            'high_cashflow_flag_acct_total'
                        ]
                        hw_count = np.zeros(len(df))
                        for ind in high_weight_indicators:
                            if ind in df.columns:
                                hw_count += df[ind].astype(int)
                        flagged_by_indicators = (hw_count >= 2).astype(int)
                        
                        df['aml_score'] = ui_score
                        df['native_score'] = native_score
                        df['anomaly_label'] = model_flag.astype(int)
                        df['flagged_by_model'] = df['anomaly_label']
                        df['flagged_by_enhanced'] = flagged_by_enhanced
                        df['flagged_by_indicators'] = flagged_by_indicators
                        df['high_weight_indicator_count'] = hw_count.astype(int)
                        df['is_flagged'] = (
                            (df['flagged_by_enhanced'] == 1) |
                            (df['flagged_by_indicators'] == 1) |
                            (df['flagged_by_model'] == 1)
                        ).astype(int)
                        
                        st.session_state.scored_data = df.copy()
                        st.session_state.selected_model = selected_model_file
                        
                        n_flagged = int(df['is_flagged'].sum())
                        flagged_rate = (n_flagged/len(df)*100) if len(df) > 0 else 0.0
                        st.success(f"✅ Scored {len(df):,} transactions. Flagged: {n_flagged:,} ({flagged_rate:.2f}%) — {policy_description}")
                        
                        st.markdown("---")
                        st.markdown("### 📊 Model Quality Assessment")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Flagging Rate", f"{flagged_rate:.2f}%")
                            if 0.1 <= flagged_rate <= 3.0:
                                st.success("✅ Good range")
                            elif 3.0 < flagged_rate <= 5.0:
                                st.warning("⚠️ Acceptable")
                            elif flagged_rate < 0.1:
                                st.info("ℹ️ Low rate - model being conservative")
                            else:
                                st.error("❌ Check threshold")
                        with col2:
                            avg_ui = float(df['aml_score'].mean()) if 'aml_score' in df.columns else 0.0
                            st.metric("Avg Score (UI)", f"{avg_ui:.3f}")
                            if avg_ui >= 0.60:
                                st.warning("⚠️ Tail-heavy (more high scores)")
                            elif avg_ui <= 0.40:
                                st.info("ℹ️ Tail-light (conservative)")
                            else:
                                st.success("✅ Balanced")

                        with col3:
                            st.metric("Flagged Transactions", f"{n_flagged}")
                            expected_high_risk = len(df) * 0.02
                            if n_flagged <= expected_high_risk * 1.5:
                                st.success("✅ Expected")
                            else:
                                st.warning("⚠️ Many flagged")
                        with col4:
                            score_std_ui = float(df['aml_score'].std()) if 'aml_score' in df.columns else 0.0
                            st.metric("Score Spread (UI)", f"{score_std_ui:.3f}")
                            if score_std_ui > 0.15:
                                st.success("✅ Good separation")
                            else:
                                st.warning("⚠️ Low variance")
                        
                        if n_flagged > 0:
                            st.markdown("---")
                            st.markdown("#### 🔍 Flagging Methods Breakdown")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                n_enhanced = int(df['flagged_by_enhanced'].sum()) if 'flagged_by_enhanced' in df.columns else 0
                                pct_enhanced = (n_enhanced / len(df) * 100) if len(df) > 0 else 0
                                st.metric("Enhanced AML Score (≥6)", f"{n_enhanced}", delta=f"{pct_enhanced:.2f}%")
                            with col2:
                                n_indicators = int(df['flagged_by_indicators'].sum()) if 'flagged_by_indicators' in df.columns else 0
                                pct_indicators = (n_indicators / len(df) * 100) if len(df) > 0 else 0
                                st.metric("2+ High-Weight Indicators", f"{n_indicators}", delta=f"{pct_indicators:.2f}%")
                            with col3:
                                n_model = int(df['flagged_by_model'].sum()) if 'flagged_by_model' in df.columns else 0
                                pct_model = (n_model / len(df) * 100) if len(df) > 0 else 0
                                st.metric("Model (policy)", f"{n_model}", delta=f"{pct_model:.2f}%")
                            
                            st.info(f"""
                            **📊 Understanding These Numbers:**
                            
                            Methods are counted separately. A transaction can be flagged by one, two, or all three.
                            **Total unique flagged:** {n_flagged:,}
                            """)
                        
                        with st.expander("🔬 Diagnostic: Why This Flagging Rate?"):
                            st.markdown(f"""
                            **Summary**
                            - Total Transactions: {len(df):,}
                            - Flagged: {n_flagged:,} ({flagged_rate:.2f}%)
                            - Contamination (training prior): {st.session_state.config.get('contamination', 0.01)*100:.1f}%
                            - Policy: {policy_description}
                            
                            **Flagging Logic**
                            1) Enhanced AML Score ≥ 6  
                            2) 2+ high-weight indicators  
                            3) Model native score meets policy threshold
                            """)
                        
                        if n_flagged > 0:
                            st.markdown("#### 🚨 Top 5 Most Suspicious Transactions")
                            top_suspicious = df.nlargest(5, 'aml_score')
                            display_cols = ['transaction_id', 'amount', 'sender_country', 'receiver_country', 
                                            'sender_bank', 'receiver_bank', 'aml_score', 'enhanced_aml_score']
                            available_display = [c for c in display_cols if c in top_suspicious.columns]
                            st.dataframe(
                                top_suspicious[available_display].style.background_gradient(
                                    subset=['aml_score'], cmap='Reds', vmin=0, vmax=1
                                ),
                                use_container_width=True,
                                hide_index=True
                            )
                            st.markdown("#### 🔍 Why Were These Flagged?")
                            for idx, (_, row) in enumerate(top_suspicious.iterrows(), 1):
                                with st.expander(f"🚨 Transaction {idx} - Score: {row.get('aml_score', 0):.3f}"):
                                    st.markdown(f"**💰 Amount:** €{row.get('amount', 0):,.2f}")
                                    st.markdown(f"**📍 From:** {row.get('sender_country', 'Unknown')} → {row.get('receiver_country', 'Unknown')}")
                                    st.markdown(f"**🏦 Banks:** {row.get('sender_bank', 'Unknown')} → {row.get('receiver_bank', 'Unknown')}")
                                    st.markdown(f"**🚩 {t('triggered_indicators')}:**")
                                    indicators_triggered = []
                                    config = st.session_state.config
                                    indicator_names = {
                                        'high_risk_sender': f"🚨 {t('high_risk_sender_desc').format(weight=config.get('weight_high_risk_sender', 3.0))}",
                                        'high_risk_receiver': f"🚨 {t('high_risk_receiver_desc').format(weight=config.get('weight_high_risk_receiver', 3.0))}",
                                        'is_high_risk_drug_country_receiver': f"💊 {t('drug_country_desc').format(weight=config.get('weight_drug_country', 4.0))}",
                                        'near_threshold': f"📊 {t('near_threshold_desc').format(threshold=config.get('reporting_threshold', 10000), weight=config.get('weight_near_threshold', 2.5))}",
                                        'structuring_flag': f"🔀 {t('structuring_desc').format(count=config.get('structuring_min_count', 3), weight=config.get('weight_structuring', 2.0))}",
                                        'midnight_transaction': f"🌙 {t('midnight_desc').format(start=config.get('midnight_start_hour', 0), end=config.get('midnight_end_hour', 6), weight=config.get('weight_midnight', 1.5))}",
                                        'high_cashflow_flag_acct_total': f"💵 {t('high_cashflow_desc').format(percentile=config.get('high_amount_percentile', 0.99)*100, weight=config.get('weight_high_cashflow', 2.5))}",
                                        'multiple_high_risk_transfers': f"🔀 {t('multiple_high_risk_desc').format(weight=config.get('weight_multiple_high_risk', 2.0))}",
                                        'rare_bank_combo_flag': f"🏦 {t('rare_bank_desc').format(weight=config.get('weight_rare_bank', 1.5))}",
                                        'definite_fraud_flag': f"🎯 {t('definite_fraud_desc')}"
                                    }
                                    for indicator_key, indicator_description in indicator_names.items():
                                        try:
                                            if indicator_key not in row.index:
                                                continue
                                            val = row[indicator_key]
                                            val_scalar = val.iloc[0] if hasattr(val, "iloc") else val
                                            try:
                                                if val_scalar is not None and abs(float(val_scalar) - 1.0) < 1e-9:
                                                    indicators_triggered.append(f"- {indicator_description}")
                                            except (ValueError, TypeError):
                                                pass
                                        except Exception:
                                            pass
                                    if indicators_triggered:
                                        for ind in indicators_triggered:
                                            st.markdown(ind)
                                    else:
                                        st.info(t('no_indicators_triggered'))
                                    if 'enhanced_aml_score' in row:
                                        st.markdown(f"**📊 {t('enhanced_aml_score_desc').format(score=row.get('enhanced_aml_score', 0))}**")
                                    st.markdown("**💡 What This Means:**")
                                    st.info("This transaction has multiple risk factors that together indicate potential fraud or money laundering activity. The AML model identified this as suspicious based on their combination.")
                    except Exception as e:
                        st.error(f"Scoring error: {str(e)}")
    else:
        st.info(t('no_models'))
    
    if st.session_state.scored_data is not None:
        st.markdown("---")
        st.markdown("### 📥 Download Results")
        df_download = st.session_state.scored_data
        col1, col2 = st.columns(2)
        with col1:
            download_option = st.radio(
                "Select data to download:",
                options=["All Scored Transactions", "Only Flagged Transactions", "Original Data Only"],
                index=0
            )
        with col2:
            if download_option == "All Scored Transactions":
                download_df = df_download
                filename = f"aml_all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                file_size = len(df_download)
            elif download_option == "Only Flagged Transactions":
                if 'is_flagged' in df_download.columns:
                    download_df = df_download[df_download['is_flagged'] == 1]
                    filename = f"aml_flagged_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    file_size = len(download_df)
                else:
                    download_df = df_download
                    filename = f"aml_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    st.warning("⚠️ No flagged data available. Downloading all.")
                    file_size = len(df_download)
            else:
                exclude_cols = ['aml_score', 'anomaly_label', 'is_flagged', 'enhanced_aml_score']
                exclude_cols = [col for col in exclude_cols if col in df_download.columns]
                download_df = df_download.drop(columns=exclude_cols) if exclude_cols else df_download
                filename = f"aml_original_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                file_size = len(download_df)
        st.info(f"📊 {download_option}: {file_size:,} rows")
        csv = download_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name=filename,
            mime="text/csv"
        )



with tab_network:
    st.header(t("network_title"))

    if st.session_state.scored_data is None:
        st.info(t("no_data"))
    else:
        df = st.session_state.scored_data

        c1, c2, c3 = st.columns(3)
        with c1:
            score_threshold_net = st.slider(
                t("score_threshold"),
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.05,
                key="net_score_threshold",
            )

        with c2:
            display_tx = st.slider(
                "Transactions to visualize",
                min_value=50,
                max_value=1500,
                value=400,
                step=50,
            )

        with c3:
            layout = st.selectbox(
                "Layout",
                options=["flow", "spring", "kamada_kawai"],
                index=0,
            )

        if st.button("Generate Network Graph", type="primary"):
            with st.spinner("Building network graph..."):
                try:
                    fig, metrics = create_transaction_network(
                        df,
                        score_threshold=score_threshold_net,
                        display_transactions=display_tx,
                        layout_algorithm=layout,
                        seed=42,
                    )

                    st.caption(
                        f"From {metrics.get('total_transactions_in_df', 0):,} transactions, "
                        f"flagged {metrics.get('flagged_transactions_in_df', 0):,}. "
                        f"Visualizing {metrics.get('selected_total', 0):,} "
                        f"({metrics.get('selected_fraud', 0):,} fraud + {metrics.get('selected_context', 0):,} context)."
                    )

                    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

                    st.markdown("### " + t("network_metrics"))
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Nodes", metrics.get("n_nodes", 0))
                    with m2:
                        st.metric("Edges", metrics.get("n_edges", 0))
                    with m3:
                        st.metric("Fraud edges", metrics.get("num_fraud_edges", 0))
                    with m4:
                        st.metric("Components", metrics.get("n_connected_components", 0))

                    if metrics.get("top_hubs"):
                        st.markdown("#### Top Hubs")
                        st.dataframe(pd.DataFrame(metrics["top_hubs"]), use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"Network graph error: {str(e)}")


with tab_analysis:
    st.header(t('tab_analysis'))
    
    
    if st.session_state.scored_data is not None:
        df = st.session_state.scored_data

        # --- Kosovo bank analysis ---
        st.subheader(t('kosovo_analysis'))

        # Check if we even have the columns needed for bank location analysis
        required_kosovo_cols = {
            'sender_country',
            'receiver_country',
            'sender_bank',
            'receiver_bank',
            'amount',
            'aml_score',
            'anomaly_label',
        }
        missing_kosovo_cols = [c for c in required_kosovo_cols if c not in df.columns]

        if missing_kosovo_cols:
            st.info("Kosovo bank analysis is not available for this dataset because "
                    "bank country / location columns were not mapped.")
        else:
            try:
                analyzer = BankFraudAnalyzer()
                kosovo_metrics = analyzer.analyze_kosovo_banks(
                    df,
                    score_column='aml_score',
                    anomaly_column='anomaly_label',
                    min_transactions=3
                )
                if len(kosovo_metrics) > 0:
                    fig = analyzer.create_bank_chart(
                        kosovo_metrics,
                        metric='risk_score',
                        top_n=10,
                        title=t('top_banks')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("### " + t('bank_metrics'))
                    st.dataframe(
                        kosovo_metrics.style.format({
                            'total_amount': '€{:,.2f}',
                            'avg_aml_score': '{:.3f}',
                            'max_aml_score': '{:.3f}',
                            'flagged_percent': '{:.2f}%',
                            'risk_score': '{:.2f}'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info(t("no_kosovo_transactions"))
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
        
        st.markdown("---")
        st.subheader(t('fraud_patterns'))
        if 'anomaly_label' in df.columns:
            fraud_df = df[df['anomaly_label'] == 1]
            if len(fraud_df) > 0:
                col1, col2 = st.columns(2)
                with col1:
                    if 'sender_country' in fraud_df.columns:
                        sender_series = fraud_df['sender_country']
                        top_sender = sender_series.value_counts().head(10) if isinstance(sender_series, pd.Series) else pd.Series(sender_series).value_counts().head(10)
                        fig = px.bar(
                            x=top_sender.values,
                            y=top_sender.index,
                            orientation='h',
                            title="Top 10 Sender Countries (Fraud)",
                            labels={'x': 'Count', 'y': 'Country'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                with col2:
                    if 'receiver_country' in fraud_df.columns:
                        receiver_series = fraud_df['receiver_country']
                        top_receiver = receiver_series.value_counts().head(10) if isinstance(receiver_series, pd.Series) else pd.Series(receiver_series).value_counts().head(10)
                        fig = px.bar(
                            x=top_receiver.values,
                            y=top_receiver.index,
                            orientation='h',
                            title="Top 10 Receiver Countries (Fraud)",
                            labels={'x': 'Count', 'y': 'Country'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                if 'amount' in fraud_df.columns and 'aml_score' in fraud_df.columns:
                    fig = px.scatter(
                        fraud_df.head(500),
                        x='amount',
                        y='aml_score',
                        color='enhanced_aml_score' if 'enhanced_aml_score' in fraud_df.columns else None,
                        title="Amount vs AML Score (Fraud Transactions)",
                        log_x=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No fraudulent transactions detected.")
    else:
        st.info(t('no_data'))



with tab_logs:
    st.header(t('tab_logs'))
    st.markdown("### 📝 Training History")
    if st.session_state.training_log:
        log_df = pd.DataFrame(st.session_state.training_log)
        st.dataframe(log_df, use_container_width=True, hide_index=True)
    else:
        st.info("No training history available yet.")
    
    st.markdown("---")
    st.markdown("### " + t('available_models'))
    if os.path.exists(st.session_state.models_dir):
        model_files = [f for f in os.listdir(st.session_state.models_dir) if f.endswith('.pkl')]
        if model_files:
            for model_file in model_files:
                with st.expander(f"📦 {model_file}"):
                    metadata_file = os.path.join(
                        st.session_state.models_dir,
                        model_file.replace('.pkl', '_metadata.json')
                    )
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        st.json(metadata)
                    else:
                        st.info("No metadata available for this model")
        else:
            st.info(t('no_models'))

st.markdown("---")
st.caption(t("global_tip"))

