import streamlit as st

TRANSLATIONS = {
    "en": {
        "app_title": "🚨 AML Fraud Detection Dashboard",
        "app_subtitle": "Production-Level Anti-Money Lauing Detection System",
        "language": "Language",
        
        "tab_overview": "📋 Overview",
        "tab_indicators": "📊 Indicators",
        "tab_models": "🤖 Models",
        "tab_network": "🕸️ Network Graph",
        "tab_analysis": "🔍 Analysis",
        "tab_logs": "📝 Logs",
        "sidebar_footer": "© 2025 AML Detection System",

        "upload_file": "Upload Transaction Data",
        "file_info": "Upload a CSV or Excel file with transaction data",
        "auto_map": "Auto-map features",
        "mapping_confidence": "Mapping Confidence",
        "select_mapping": "Select column mappings manually",
        "data_preview": "Data Preview",
        
        "indicators_config": "Indicator Configuration",
        "reporting_threshold": "Reporting Threshold (Cash)",
        "near_threshold_buffer": "Near Threshold Buffer",
        "structuring_min": "Structuring Min Count",
        "multi_bank_min": "Multi-Bank Min Count",
        "contamination": "Contamination Rate",
        "high_amount_percentile": "High Amount Percentile",
        "midnight_start_hour": "Midnight Transaction Start Hour",
        "midnight_end_hour": "Midnight Transaction End Hour",
        "midnight_start_help": "Start hour for midnight detection (0 = midnight)",
        "midnight_end_help": "End hour for midnight detection (6 = 6 AM)",
        "high_risk_sender_weight": "High-Risk Sender Weight",
        "high_risk_receiver_weight": "High-Risk Receiver Weight",
        "drug_country_weight": "Drug Country Weight",
        "near_threshold_weight": "Near Threshold Weight",
        "structuring_weight": "Structuring Weight",
        "midnight_weight": "Midnight Transaction Weight",
        "high_cashflow_weight": "High Cashflow Weight",
        "multiple_high_risk_weight": "Multiple High-Risk Transfers Weight",
        "rare_bank_combo_weight": "Rare Bank Combo Weight",
        "high_risk_sender_help": "Weight for high-risk sender country flag",
        "high_risk_receiver_help": "Weight for high-risk receiver country flag",
        "drug_country_help": "Weight for drug trafficking country flag",
        "high_cashflow_help": "Weight for high cashflow transactions",
        "rare_bank_help": "Weight for rare bank combinations",
        "triggered_indicators": "Triggered Risk Indicators",
        "high_risk_sender_desc": "High-Risk Sender Country (Weight: {weight})",
        "high_risk_receiver_desc": "High-Risk Receiver Country (Weight: {weight})",
        "drug_country_desc": "Drug Trafficking Country (Weight: {weight})",
        "near_threshold_desc": "Near Threshold {threshold:,.0f} (Weight: {weight})",
        "structuring_desc": "Structuring Pattern - {count}+ small transactions (Weight: {weight})",
        "midnight_desc": "Midnight Transaction {start}-{end} AM (Weight: {weight})",
        "high_cashflow_desc": "High Cashflow - Top {percentile:.0f}% (Weight: {weight})",
        "multiple_high_risk_desc": "Multiple High-Risk Transfers (Weight: {weight})",
        "rare_bank_desc": "Rare Bank Combination (Weight: {weight})",
        "definite_fraud_desc": "DEFINITE FRAUD FLAG (Multiple conditions met)",
        "no_indicators_triggered": "No specific indicators triggered, but transaction shows unusual pattern overall.",
        "enhanced_aml_score_desc": "Enhanced AML Score: {score:.1f} / ~30 (higher = more indicators triggered)",
        "core_thresholds": "Core Thresholds",
        "pattern_detection": "Pattern Detection",
        "risk_country_lists": "Risk Country Lists",
        "high_risk_countries": "High-Risk Countries (29 countries)",
        "drug_trafficking_countries": "Drug Trafficking Countries (9 countries)",
        "feature_weights": "Feature Weights (Enhanced AML Score)",
        "save_config": "💾 Save Configuration",
        "config_saved": "Configuration saved successfully",
        "adjust_mappings_expander": "🔧 Adjust column mappings manually",
        "currently_mapped": "**Currently mapped:** {mapped_count} out of {total_features} features",
        "required_fields_info": "🔴 **Required fields:** {required} ({mapped}/{total_required} mapped)",
        "auto_mapping_tip": "💡 **Tip:** Auto-mapping has detected all available features. You can adjust or remove any mappings below.",
        "apply_manual_mapping": "✅ Apply Manual Mapping",
        "manual_mapping_success": "✅ **Manual mapping applied successfully!**",
        "manual_mapping_summary": "**Mapped {mapped}/{total} features** | Required fields: {required_mapped}/{total_required}",
        "mapping_error_prefix": "Mapping error",
        "processing_data_spinner": "Processing data...",
        "processing_error_prefix": "Processing error",
        "process_data_button": "🔄 Process Data",
        "process_data_missing_required_title": "⚠️ **Cannot process data: Missing required fields**",
        "process_data_missing_required_list": "The following required fields must be mapped: **{missing}**",
        "process_data_missing_required_tip": "💡 Please use the manual mapping section above to map these required fields.",
        "process_data_success_mapped": "✅ **Mapped {n_cols} columns from your data:** {cols}",
        "process_data_success_done": "✅ **Data processing complete!**",
        
        "train_new_model": "Train New Models",
        "select_algorithms": "Select Algorithms",
        "train_button": "🚀 Train Models",
        "retrain_button": "🔄 Retrain with New Data",
        "score_button": "📊 Score Transactions",
        "model_selection": "Select Model",
        "threshold_slider": "Alert Threshold",
        "available_models": "Available Models",
        
        "total_transactions": "Total Transactions",
        "flagged_transactions": "Flagged Transactions",
        "flagged_rate": "Flagged Rate",
        "avg_score": "Avg AML Score",
        "download_results": "⬇️ Download Results",
        
        "network_title": "Transaction Network Graph",
        "network_filters": "Graph Filters",
        "score_threshold": "Min AML Score",
        "max_transactions": "Max Transactions to Display",
        "layout_algorithm": "Layout Algorithm",
        "network_metrics": "Network Metrics",
        
        "kosovo_analysis": "Kosovo Bank Fraud Analysis",
        "top_banks": "Top 10 Most Fraudulent Banks",
        "bank_metrics": "Bank Fraud Metrics",
        "fraud_patterns": "Fraud Pattern Analysis",
        "kosovo_analysis_not_available": (
            "Kosovo bank analysis is not available for this dataset because "
            "bank country / location columns were not mapped."
        ),

        
        "processing": "Processing...",
        "training": "Training models...",
        "success": "Success!",
        "error": "Error",
        "no_data": "No data available. Please upload data first.",
        "no_models": "No models available. Please train a model first.",

        # === New general / UI messages ===
        "global_tip": "💡 Tip: This is a production-level AML detection system. Adjust indicators, train models, and analyze patterns to optimize fraud detection.",
        "indicators_config_warning": "⚠️ Configure these BEFORE processing data in Overview tab",
        "load_success": "✅ Loaded {n:,} transactions",
        "kosovo_analysis_not_available": "Kosovo bank analysis is not available for this dataset because bank location columns were not mapped.",

        # === Models tab / scoring & diagnostics ===
        "score_transactions_title": "Score Transactions",
        "alert_policy": "Alert policy",
        "alert_policy_native": "Native (from training)",
        "alert_policy_manual": "Manual cutoff",
        "alert_policy_topk": "Top-k% (capacity)",
        "manual_cutoff_help": "Manual cutoff (model-native units)",
        "native_threshold_label": "Native threshold (from training): {threshold:.4f}",
        "target_alert_rate": "Target alert rate (% of txns)",

        "training_summary": "✅ Trained {n_models} model(s)",
        "scoring_summary": "✅ Scored {n_txns:,} transactions. Flagged: {n_flagged:,} ({flagged_rate:.2f}%) — Native threshold = {threshold:.6f}",

        "model_quality_title": "📊 Model Quality Assessment",
        "flagging_rate": "Flagging Rate",
        "good_range": "✅ Good range",
        "avg_score_ui": "Avg Score (UI)",
        "balanced": "✅ Balanced",
        "expected_ok": "✅ Expected",
        "score_spread_ui": "Score Spread (UI)",
        "good_separation": "✅ Good separation",

        "flagging_methods_breakdown": "🔍 Flagging Methods Breakdown",
        "flagging_methods_enhanced_aml": "Enhanced AML Score (≥6)",
        "flagging_methods_two_plus_indicators": "2+ High-Weight Indicators",
        "flagging_methods_model_policy": "Model (policy)",

        "understanding_numbers_title": "📊 Understanding These Numbers:",
        "understanding_numbers_explanation": "Methods are counted separately. A transaction can be flagged by one, two, or all three. Total unique flagged: {total_unique}",

        "diagnostic_flagging_rate": "🔬 Diagnostic: Why This Flagging Rate?",
        "top_suspicious_transactions": "🚨 Top 5 Most Suspicious Transactions",
        "why_flagged_title": "🔍 Why Were These Flagged?",

        "download_results_title": "📥 Download Results",
        "download_select_label": "Select data to download:",
        "download_option_all_scored": "All Scored Transactions",
        "download_option_only_flagged": "Only Flagged Transactions",
        "download_option_original_only": "Original Data Only",
        "all_scored_transactions_summary": "📊 All Scored Transactions: {rows:,} rows",

        # === Network tab extras ===
        "generate_network_graph": "🔄 Generate Network Graph",
        "network_nodes": "Nodes",
        "network_edges": "Edges",
        "network_density": "Density",
        "network_components": "Components",
        "network_top_hubs": "Top Hubs (Most Connected Accounts)",
        "network_debug_expander": "🔍 Debug: Available Columns",

        # === Bank analysis tab ===
        "top_sender_countries_fraud": "Top 10 Sender Countries (Fraud)",
        "top_receiver_countries_fraud": "Top 10 Receiver Countries (Fraud)",
        "amount_vs_aml_fraud": "Amount vs AML Score (Fraud Transactions)",
        "count": "Count",
        "country": "Country",
        "no_fraud_transactions": "No fraudulent transactions detected.",
        "no_kosovo_transactions": "No Kosovo-related transactions found in the dataset.",

        # === Logs / training history ===
        "training_history": "📝 Training History",
    },

    "al": {
        "app_title": "🚨 Paneli i Zbulimit të Mashtrimit AML",
        "app_subtitle": "Sistemi i Avancuar për Zbulimin e Pastrimit të Parave",
        "language": "Gjuha",
        "sidebar_footer": "© 2025 Sistemi i Zbulimit AML",
        "tab_overview": "📋 Përmbledhje",
        "tab_indicators": "📊 Treguesit",
        "tab_models": "🤖 Modelet",
        "tab_network": "🕸️ Rrjeti Grafik",
        "tab_analysis": "🔍 Analiza",
        "tab_logs": "📝 Historiku",
        
        "upload_file": "Ngarko të Dhënat e Transaksioneve",
        "file_info": "Ngarko një skedar CSV ose Excel me të dhënat e transaksioneve",
        "auto_map": "Hartëzimi automatik i kolonave",
        "mapping_confidence": "Besueshmëria e Hartëzimit",
        "select_mapping": "Zgjidhni hartëzimin e kolonave manualisht",
        "data_preview": "Pamja Paraprake e të Dhënave",
        
        "indicators_config": "Konfigurimi i Treguesve",
        "reporting_threshold": "Pragu i Raportimit (Cash)",
        "near_threshold_buffer": "Buffer afër Pragut",
        "structuring_min": "Numri Min. i Strukturimit",
        "multi_bank_min": "Numri Min. i Bankave",
        "contamination": "Niveli i Kontaminimit",
        "high_amount_percentile": "Përqindja e Shumës së Lartë",
        "midnight_start_hour": "Ora e Fillimit të Transaksionit të Mesnatës",
        "midnight_end_hour": "Ora e Mbarimit të Transaksionit të Mesnatës",
        "midnight_start_help": "Ora e fillimit për zbulimin e mesnatës (0 = mesnatë)",
        "midnight_end_help": "Ora e mbarimit për zbulimin e mesnatës (6 e mëngjesit)",
        "high_risk_sender_weight": "Pesha e Dërguesit me Rrezik të Lartë",
        "high_risk_receiver_weight": "Pesha e Përfituesit me Rrezik të Lartë",
        "drug_country_weight": "Pesha e Shtetit të Trafikimit të Droges",
        "near_threshold_weight": "Pesha afër Pragut",
        "structuring_weight": "Pesha e Strukturimit",
        "midnight_weight": "Pesha e Transaksionit të Mesnatës",
        "high_cashflow_weight": "Pesha e Rrjedhjes së Lartë të Cash",
        "multiple_high_risk_weight": "Pesha e Transferave të Shumëfishta me Rrezik të Lartë",
        "rare_bank_combo_weight": "Pesha e Kombinimit të Rralë të Bankave",
        "high_risk_sender_help": "Pesha për flamurin e shtetit dërgues me rrezik të lartë",
        "high_risk_receiver_help": "Pesha për flamurin e shtetit përfitues me rrezik të lartë",
        "drug_country_help": "Pesha për flamurin e shtetit të trafikimit të drogës",
        "high_cashflow_help": "Pesha për transaksionet me rrjedhje të lartë të cash",
        "rare_bank_help": "Pesha për kombinimet e rralla të bankave",
        "triggered_indicators": "Treguesit e Rrezikut të Aktivizuar",
        "high_risk_sender_desc": "Shteti Dërgues me Rrezik të Lartë (Pesha: {weight})",
        "high_risk_receiver_desc": "Shteti Përfitues me Rrezik të Lartë (Pesha: {weight})",
        "drug_country_desc": "Shteti i Trafikimit të Droges (Pesha: {weight})",
        "near_threshold_desc": "Afër Pragut {threshold:,.0f} (Pesha: {weight})",
        "structuring_desc": "Strukturim - {count}+ transaksione të vogla (Pesha: {weight})",
        "midnight_desc": "Transaksion Mesnate {start}-{end} (Pesha: {weight})",
        "high_cashflow_desc": "Rrjedhje e Lartë e Cash - Top {percentile:.0f}% (Pesha: {weight})",
        "multiple_high_risk_desc": "Transfera të Shumëfishta me Rrezik të Lartë (Pesha: {weight})",
        "rare_bank_desc": "Kombinim i Rralë i Bankave (Pesha: {weight})",
        "definite_fraud_desc": "FLAMUR I SIGURT I MASHTRIMIT (Kushte të shumta të përmbushura)",
        "no_indicators_triggered": "Asnjë tregues specifik i aktivizuar, por transaksioni tregon një model të pazakontë në përgjithësi.",
        "enhanced_aml_score_desc": "Rezultati i Përmirësuar AML: {score:.1f} / ~30 (më i lartë = më shumë tregues të aktivizuar)",
        "core_thresholds": "Pragjet Themelore",
        "pattern_detection": "Zbulimi i Modeleve",
        "risk_country_lists": "Listat e Shteteve me Rrezik",
        "high_risk_countries": "Shtetet me Rrezik të Lartë (29 shtete)",
        "drug_trafficking_countries": "Shtetet e Trafikimit të Droges (9 shtete)",
        "feature_weights": "Peshat e Karakteristikave (Rezultati i Përmirësuar AML)",
        "save_config": "💾 Ruaj Konfigurimin",
        "config_saved": "Konfigurimi u ruajt me sukses",
        "adjust_mappings_expander": "🔧 Rregulloni manualisht hartëzimin e kolonave",
        "currently_mapped": "**Tani të hartëzuara:** {mapped_count} nga {total_features} veçori",
        "required_fields_info": "🔴 **Fushat e detyrueshme:** {required} ({mapped}/{total_required} të hartëzuara)",
        "auto_mapping_tip": "💡 **Këshillë:** Hartëzimi automatik ka identifikuar të gjitha veçoritë e disponueshme. Mund t’i ndryshoni ose hiqni më poshtë.",
        "apply_manual_mapping": "✅ Apliko Hartëzimin Manual",
        "manual_mapping_success": "✅ **Hartëzimi manual u aplikua me sukses!**",
        "manual_mapping_summary": "**Të hartëzuara {mapped}/{total} veçori** | Fushat e detyrueshme: {required_mapped}/{total_required}",
        "mapping_error_prefix": "Gabim në hartëzim",
        "processing_data_spinner": "Duke përpunuar të dhënat...",
        "processing_error_prefix": "Gabim gjatë përpunimit",
        "process_data_button": "🔄 Përpuno të Dhënat",
        "process_data_missing_required_title": "⚠️ **Nuk mund të përpunohen të dhënat: mungojnë fusha të detyrueshme**",
        "process_data_missing_required_list": "Fushat e detyrueshme që duhen hartëzuar: **{missing}**",
        "process_data_missing_required_tip": "💡 Ju lutem përdorni seksionin e hartëzimit manual për t’i plotësuar këto fusha.",
        "process_data_success_mapped": "✅ **U hartëzuan {n_cols} kolona nga të dhënat tuaja:** {cols}",
        "process_data_success_done": "✅ **Përpunimi i të dhënave përfundoi me sukses!**",
        
        "train_new_model": "Trajno Modele të Reja",
        "select_algorithms": "Zgjidhni Algoritmet",
        "train_button": "🚀 Trajno Modelet",
        "retrain_button": "🔄 Ri-trajno me të Dhëna të Reja",
        "score_button": "📊 Vlerëso Transaksionet",
        "model_selection": "Zgjidhni Modelin",
        "threshold_slider": "Pragu i Alarmit",
        "available_models": "Modelet e Disponueshme",
        
        "total_transactions": "Totali i Transaksioneve",
        "flagged_transactions": "Transaksionet e Shënuara",
        "flagged_rate": "Shkalla e Shenimit",
        "avg_score": "Rezultati Mesatar AML",
        "download_results": "⬇️ Shkarko Rezultatet",
        
        "network_title": "Grafiku i Rrjetit të Transaksioneve",
        "network_filters": "Filtrat e Grafikut",
        "score_threshold": "Rezultati Min AML",
        "max_transactions": "Maks. Transaksionet për të Shfaqur",
        "layout_algorithm": "Algoritmi i Paraqitjes",
        "network_metrics": "Metrikat e Rrjetit",
        
        "kosovo_analysis": "Analiza e Mashtrimit të Bankave të Kosovës",
        "top_banks": "Top 10 Bankat më të Dyshuara",
        "bank_metrics": "Metrikat e Mashtrimit të Bankave",
        "fraud_patterns": "Analiza e Modeleve të Mashtrimit",
        "kosovo_analysis_not_available": (
            "Analiza e bankave të Kosovës nuk është e disponueshme për këtë set të dhënash "
            "sepse kolonat e vendndodhjes së bankave nuk janë hartëzuar." 
            ),
        "no_kosovo_transactions": "Nuk u gjetën transaksione të lidhura me Kosovën në këtë set të dhënash.",
        
        "processing": "Duke përpunuar...",
        "training": "Duke trajnuar modelet...",
        "success": "Sukses!",
        "error": "Gabim",
        "no_data": "Nuk ka të dhëna. Ju lutem ngarkoni të dhënat fillimisht.",
        "no_models": "Nuk ka modele. Ju lutem trajnoni një model fillimisht.",

        # === New general / UI messages ===
        "global_tip": "💡 Këshillë: Ky është një sistem prodhimi për zbulimin e pastrimit të parave. Rregulloni treguesit, trajnojeni modelin dhe analizoni modelet për të optimizuar zbulimin e mashtrimit.",
        "indicators_config_warning": "⚠️ Konfigurojini këto PËRPARA se të përpunoni të dhënat në tabin Përmbledhje",
        "load_success": "✅ U ngarkuan {n:,} transaksione",
        "kosovo_analysis_not_available": "Analiza e bankave në Kosovë nuk është e disponueshme për këtë set të dhënash sepse kolonat e vendndodhjes së bankave nuk janë hartëzuar.",

        # === Models tab / scoring & diagnostics ===
        "score_transactions_title": "Vlerëso Transaksionet",
        "alert_policy": "Politika e alarmit",
        "alert_policy_native": "Nativ (nga trajnimi)",
        "alert_policy_manual": "Prerje manuale",
        "alert_policy_topk": "Top-k% (kapaciteti)",
        "manual_cutoff_help": "Prag manual (njësi të modelit)",
        "native_threshold_label": "Pragu natyral (nga trajnimi): {threshold:.4f}",
        "target_alert_rate": "Shkalla e synuar e alarmeve (% e transaksioneve)",

        "training_summary": "✅ U trajnuan {n_models} model(e)",
        "scoring_summary": "✅ U vlerësuan {n_txns:,} transaksione. Të shënuara: {n_flagged:,} ({flagged_rate:.2f}%) — Pragu natyral = {threshold:.6f}",

        "model_quality_title": "📊 Vlerësimi i Cilësisë së Modelit",
        "flagging_rate": "Shkalla e shënimit",
        "good_range": "✅ Interval i mirë",
        "avg_score_ui": "Rezultati mesatar (UI)",
        "balanced": "✅ I balancuar",
        "expected_ok": "✅ Siç pritej",
        "score_spread_ui": "Shpërndarja e rezultatit (UI)",
        "good_separation": "✅ Ndarje e mirë",

        "flagging_methods_breakdown": "🔍 Ndarja sipas metodave të shënimit",
        "flagging_methods_enhanced_aml": "Rezultati i përmirësuar AML (≥6)",
        "flagging_methods_two_plus_indicators": "2+ tregues me peshë të lartë",
        "flagging_methods_model_policy": "Modeli (politika)",

        "understanding_numbers_title": "📊 Kuptimi i këtyre vlerave:",
        "understanding_numbers_explanation": "Metodat numërohen veçmas. Një transaksion mund të shënohet nga një, dy ose të treja. Numri total unik i transaksioneve të shënuara: {total_unique}",

        "diagnostic_flagging_rate": "🔬 Diagnostikë: Pse kjo shkallë shënimi?",
        "top_suspicious_transactions": "🚨 5 transaksionet më të dyshimta",
        "why_flagged_title": "🔍 Pse u shënuan këto?",

        "download_results_title": "📥 Shkarko rezultatet",
        "download_select_label": "Zgjidhni të dhënat për shkarkim:",
        "download_option_all_scored": "Të gjitha transaksionet e vlerësuara",
        "download_option_only_flagged": "Vetëm transaksionet e shënuara",
        "download_option_original_only": "Vetëm të dhënat origjinale",
        "all_scored_transactions_summary": "📊 Të gjitha transaksionet e vlerësuara: {rows:,} rreshta",

        # === Network tab extras ===
        "generate_network_graph": "🔄 Gjenero grafikun e rrjetit",
        "network_nodes": "Nyje",
        "network_edges": "Lidhje",
        "network_density": "Dendësia",
        "network_components": "Komponentë",
        "network_top_hubs": "Nyjet kryesore (llogaritë më të lidhura)",
        "network_debug_expander": "🔍 Diagnostikë: Kolonat në dispozicion",

        # === Bank analysis tab ===
        "top_sender_countries_fraud": "Top 10 vendet e dërguesve (transaksione të dyshimta)",
        "top_receiver_countries_fraud": "Top 10 vendet e përfituesve (transaksione të dyshimta)",
        "amount_vs_aml_fraud": "Shuma kundrejt rezultatit AML (transaksione të dyshimta)",
        "count": "Numri",
        "country": "Vendi",
        "no_fraud_transactions": "Nuk u gjetën transaksione të dyshimta.",

        # === Logs / training history ===
        "training_history": "📝 Historia e Trajnimit",
    }
}

DEFAULT_LANG = "en"


def t(key: str, **kwargs) -> str:
    
    lang = st.session_state.get("language", DEFAULT_LANG)

    lang_dict = TRANSLATIONS.get(lang, TRANSLATIONS[DEFAULT_LANG])
    text = lang_dict.get(key, TRANSLATIONS[DEFAULT_LANG].get(key, key))

    if kwargs:
        try:
            text = text.format(**kwargs)
        except Exception:
            pass

    return text
