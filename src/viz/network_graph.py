from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import math
import random

import pandas as pd
import networkx as nx
import plotly.graph_objects as go


def _guess_col(df: pd.DataFrame, direct: List[str], keywords: List[str]) -> Optional[str]:
    for c in direct:
        if c in df.columns:
            return c
    for c in df.columns:
        cl = str(c).lower()
        if any(k in cl for k in keywords):
            return c
    return None


def select_transactions_for_network(
    df: pd.DataFrame,
    score_threshold: float,
    display_transactions: int,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Professional selection:
    - Include fraud transactions first (score >= threshold)
    - Fill remaining slots with context transactions that share accounts with fraud
    - If still not enough, fill with highest-score normal transactions
    """
    if df is None or df.empty or display_transactions <= 0:
        return df.head(0), {"selected_total": 0, "selected_fraud": 0, "selected_context": 0}

    tmp = df.copy()

    score_col = _guess_col(
        tmp,
        direct=["enhanced_aml_score", "anomaly_score", "aml_score", "score"],
        keywords=["aml", "anomaly", "score"],
    )

    if score_col:
        tmp["_score_num"] = pd.to_numeric(tmp[score_col], errors="coerce").fillna(0.0)
    else:
        tmp["_score_num"] = 0.0

    sender_col = _guess_col(
        tmp,
        direct=["sender_id", "id_i_llogarise_se_derguesit", "sender_account", "from_account"],
        keywords=["sender", "from", "dergues", "llogarise_se_derguesit"],
    )
    receiver_col = _guess_col(
        tmp,
        direct=["receiver_id", "id_i_llogarise_se_perfituesit", "receiver_account", "to_account"],
        keywords=["receiver", "to", "perfitues", "llogarise_se_perfituesit"],
    )
    amt_col = _guess_col(
        tmp,
        direct=["amount", "transaction_amount", "shuma", "value", "amt"],
        keywords=["amount", "shuma", "amt", "value"],
    )

    fraud_df = tmp[tmp["_score_num"] >= float(score_threshold)].copy()
    normal_df = tmp[tmp["_score_num"] < float(score_threshold)].copy()

    # If fraud > display limit: show top fraud only
    if len(fraud_df) >= display_transactions:
        selected = fraud_df.sort_values("_score_num", ascending=False).head(display_transactions)
        selected = selected.drop(columns=["_score_num"], errors="ignore")
        return selected, {
            "selected_total": int(len(selected)),
            "selected_fraud": int(len(selected)),
            "selected_context": 0,
        }

    # Otherwise include all fraud and fill with context
    selected = fraud_df
    remaining = display_transactions - len(selected)

    context = pd.DataFrame(columns=tmp.columns)

    if remaining > 0 and sender_col and receiver_col and len(fraud_df) > 0:
        fraud_nodes = set(fraud_df[sender_col].astype(str)) | set(fraud_df[receiver_col].astype(str))

        cand = normal_df[
            normal_df[sender_col].astype(str).isin(fraud_nodes)
            | normal_df[receiver_col].astype(str).isin(fraud_nodes)
        ].copy()

        if amt_col and amt_col in cand.columns:
            cand["_amt_num"] = pd.to_numeric(cand[amt_col], errors="coerce").fillna(0.0)
            cand = cand.sort_values(["_score_num", "_amt_num"], ascending=[False, False])
        else:
            cand = cand.sort_values(["_score_num"], ascending=[False])

        context = cand.head(remaining)

    # still not enough -> fill with highest-score normal
    if remaining > 0 and len(context) < remaining:
        still = remaining - len(context)
        filler = normal_df.sort_values("_score_num", ascending=False).head(still)
        context = pd.concat([context, filler], ignore_index=True)

    final_df = pd.concat([selected, context], ignore_index=True)

    counts = {
        "selected_total": int(len(final_df)),
        "selected_fraud": int(len(selected)),
        "selected_context": int(len(final_df) - len(selected)),
    }

    final_df = final_df.drop(columns=["_score_num", "_amt_num"], errors="ignore")
    return final_df, counts


class TransactionNetworkGraph:
    """
    Directed account-to-account transaction network.
    Nodes: accounts.
    Edges: aggregated sender->receiver relationship (multiple tx possible).
    """

    MAX_TX_PER_EDGE: int = 6

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self.pos: Dict[Any, Tuple[float, float]] = {}
        self.score_threshold: float = 0.5
        self.score_column_used: Optional[str] = None

    # -----------------------------
    # Column helpers
    # -----------------------------
    @staticmethod
    def _find_column_by_keywords(columns: List[str], keywords: List[str]) -> Optional[str]:
        keywords_lower = [k.lower() for k in keywords]
        for col in columns:
            col_lower = str(col).lower()
            for keyword in keywords_lower:
                if keyword in col_lower:
                    return col
        return None

    @staticmethod
    def _get_column_value(
        row: pd.Series,
        all_columns: List[str],
        possible_names: List[str],
        default: str = "Unknown",
    ) -> str:
        columns_lower = {str(col).lower(): col for col in all_columns}

        for name in possible_names:
            if name in all_columns:
                value = row[name]
                if not pd.isna(value) and str(value).strip() != "":
                    return str(value).strip()

            name_lower = name.lower()
            if name_lower in columns_lower:
                actual_col = columns_lower[name_lower]
                value = row[actual_col]
                if not pd.isna(value) and str(value).strip() != "":
                    return str(value).strip()

        col = TransactionNetworkGraph._find_column_by_keywords(all_columns, possible_names)
        if col:
            value = row[col]
            if not pd.isna(value) and str(value).strip() != "":
                return str(value).strip()

        return default

    @staticmethod
    def _get_column_value_raw(
        row: pd.Series,
        all_columns: List[str],
        possible_names: List[str],
        default: Any = None,
    ) -> Any:
        columns_lower = {str(col).lower(): col for col in all_columns}

        for name in possible_names:
            if name in all_columns:
                value = row[name]
                if not pd.isna(value):
                    return value

            name_lower = name.lower()
            if name_lower in columns_lower:
                actual_col = columns_lower[name_lower]
                value = row[actual_col]
                if not pd.isna(value):
                    return value

        col = TransactionNetworkGraph._find_column_by_keywords(all_columns, possible_names)
        if col:
            value = row[col]
            if not pd.isna(value):
                return value

        return default

    @staticmethod
    def detect_score_column(df: pd.DataFrame) -> Optional[str]:
        for c in ["enhanced_aml_score", "anomaly_score", "aml_score", "score"]:
            if c in df.columns:
                return c
        return None

    # -----------------------------
    # Build graph (from already-selected df)
    # -----------------------------
    def build_graph(
        self,
        df: pd.DataFrame,
        score_threshold: Optional[float] = None,
    ) -> nx.DiGraph:
        G = nx.DiGraph()

        if df is None or df.empty:
            self.graph = G
            self.pos = {}
            self.score_column_used = None
            return G

        try:
            self.score_threshold = float(score_threshold) if score_threshold is not None else 0.5
        except (TypeError, ValueError):
            self.score_threshold = 0.5

        self.score_column_used = self.detect_score_column(df)
        all_columns = df.columns.tolist()

        sender_keys = ["sender_id", "id_i_llogarise_se_derguesit", "sender_account", "from_account", "sender"]
        receiver_keys = ["receiver_id", "id_i_llogarise_se_perfituesit", "receiver_account", "to_account", "receiver"]
        amount_keys = ["amount", "shuma", "transaction_amount", "value", "amt"]
        sender_bank_keys = ["sender_bank", "banka_e_derguesit", "bank_sender", "from_bank", "banka_derguese"]
        receiver_bank_keys = ["receiver_bank", "banka_e_perfituesit", "bank_receiver", "to_bank", "banka_pranuese"]
        txid_keys = ["transaction_id", "id_transaksioni", "id", "tx_id"]
        ts_keys = ["timestamp", "date", "transaction_date", "created_at"]

        for idx, row in df.iterrows():
            sender_raw = self._get_column_value_raw(row, all_columns, sender_keys)
            receiver_raw = self._get_column_value_raw(row, all_columns, receiver_keys)

            sender = str(sender_raw).strip() if sender_raw is not None and str(sender_raw).strip() else f"S_{idx}"
            receiver = str(receiver_raw).strip() if receiver_raw is not None and str(receiver_raw).strip() else f"R_{idx}"

            amount_raw = self._get_column_value_raw(row, all_columns, amount_keys, default=0.0)
            try:
                amount = float(amount_raw) if amount_raw not in (None, "") else 0.0
            except (TypeError, ValueError):
                amount = 0.0

            tx_id = self._get_column_value(row, all_columns, txid_keys, default=str(idx))
            timestamp = self._get_column_value_raw(row, all_columns, ts_keys, default=None)

            sender_bank = self._get_column_value(row, all_columns, sender_bank_keys, default="Unknown")
            receiver_bank = self._get_column_value(row, all_columns, receiver_bank_keys, default="Unknown")

            score_val = 0.0
            if self.score_column_used and self.score_column_used in df.columns:
                s_raw = row.get(self.score_column_used)
                try:
                    score_val = float(s_raw) if s_raw not in (None, "") and not pd.isna(s_raw) else 0.0
                except (TypeError, ValueError):
                    score_val = 0.0

            # nodes
            if sender not in G:
                G.add_node(sender)
            if receiver not in G:
                G.add_node(receiver)

            tx_record = {
                "transaction_id": tx_id,
                "amount": amount,
                "sender_bank": sender_bank,
                "receiver_bank": receiver_bank,
                "score": score_val,
                "timestamp": timestamp,
            }

            if G.has_edge(sender, receiver):
                e = G[sender][receiver]

                w = e.get("weight", 0)
                try:
                    w_int = int(w)
                except (TypeError, ValueError):
                    w_int = 0
                e["weight"] = w_int + 1

                old_total = e.get("total_amount", 0.0)
                try:
                    old_total_f = float(old_total)
                except (TypeError, ValueError):
                    old_total_f = 0.0
                e["total_amount"] = old_total_f + amount

                old_max = e.get("max_score", 0.0)
                try:
                    old_max_f = float(old_max)
                except (TypeError, ValueError):
                    old_max_f = 0.0
                e["max_score"] = max(old_max_f, score_val)

                tx_list = e.get("tx_list", [])
                if not isinstance(tx_list, list):
                    tx_list = []
                tx_list.append(tx_record)
                e["tx_list"] = tx_list[-self.MAX_TX_PER_EDGE :]
            else:
                G.add_edge(
                    sender,
                    receiver,
                    weight=1,
                    total_amount=amount,
                    max_score=score_val,
                    tx_list=[tx_record],
                )

        self.graph = G
        self.pos = {}
        return G

    # -----------------------------
    # Professional layouts
    # -----------------------------
    def _flow_layout(self, seed: int = 42) -> Dict[Any, Tuple[float, float]]:
        """
        Left-to-right layout:
        - nodes with net outflow (out > in) pushed left
        - net inflow (in > out) pushed right
        - neutral in the middle
        """
        G = self.graph
        if len(G) == 0:
            return {}

        rnd = random.Random(seed)

        # compute in/out degree counts without DegreeView typing issues
        in_deg_map: Dict[Any, int] = {n: 0 for n in G.nodes}
        out_deg_map: Dict[Any, int] = {n: 0 for n in G.nodes}
        for u, v in G.edges:
            out_deg_map[u] = out_deg_map.get(u, 0) + 1
            in_deg_map[v] = in_deg_map.get(v, 0) + 1

        left: List[Any] = []
        mid: List[Any] = []
        right: List[Any] = []

        for n in G.nodes:
            net = out_deg_map.get(n, 0) - in_deg_map.get(n, 0)
            if net > 0:
                left.append(n)
            elif net < 0:
                right.append(n)
            else:
                mid.append(n)

        def sort_key(n: Any) -> int:
            return (in_deg_map.get(n, 0) + out_deg_map.get(n, 0))

        left.sort(key=sort_key, reverse=True)
        mid.sort(key=sort_key, reverse=True)
        right.sort(key=sort_key, reverse=True)

        pos: Dict[Any, Tuple[float, float]] = {}

        def place(nodes: List[Any], x: float) -> None:
            m = len(nodes)
            for i, n in enumerate(nodes):
                y = 0.0 if m <= 1 else (i / (m - 1)) * 2.0 - 1.0
                pos[n] = (
                    x + (rnd.random() - 0.5) * 0.06,
                    y + (rnd.random() - 0.5) * 0.06,
                )

        place(left, -1.0)
        place(mid, 0.0)
        place(right, 1.0)

        return pos

    def _compute_layout(self, algorithm: str = "flow", seed: int = 42) -> Dict[Any, Tuple[float, float]]:
        if len(self.graph) == 0:
            self.pos = {}
            return {}

        if algorithm == "flow":
            raw_pos = self._flow_layout(seed=seed)
        elif algorithm == "kamada_kawai":
            raw_pos = nx.kamada_kawai_layout(self.graph)
        else:
            # spring (stable + readable)
            n = max(len(self.graph), 1)
            k = 1.2 / (n ** 0.5)
            raw_pos = nx.spring_layout(self.graph, seed=seed, k=k, iterations=250)

        self.pos = {node: (float(coords[0]), float(coords[1])) for node, coords in raw_pos.items()}
        return self.pos

    # -----------------------------
    # Plotly rendering
    # -----------------------------
    def _create_edge_traces(self) -> List[go.Scatter]:
        if not self.pos:
            self._compute_layout("flow", seed=42)

        try:
            thr = float(self.score_threshold)
        except (TypeError, ValueError):
            thr = 0.5

        traces: List[go.Scatter] = []

        for sender, receiver, data in self.graph.edges(data=True):
            x0, y0 = self.pos[sender]
            x1, y1 = self.pos[receiver]

            raw_max = data.get("max_score", 0.0)
            try:
                max_score = float(raw_max) if raw_max not in (None, "") else 0.0
            except (TypeError, ValueError):
                max_score = 0.0

            is_fraud = max_score >= thr
            color = "rgba(200, 60, 60, 0.92)" if is_fraud else "rgba(140, 140, 140, 0.18)"

            w_raw = data.get("weight", 1)
            try:
                w_val = float(w_raw) if w_raw not in (None, "") else 1.0
            except (TypeError, ValueError):
                w_val = 1.0

            width = 0.7 + 1.3 * math.log1p(max(w_val, 1.0))
            width = min(width, 6.0)
            if is_fraud:
                width = min(width * 1.25, 7.0)

            tx_list = data.get("tx_list", [])
            if not isinstance(tx_list, list):
                tx_list = []
            tx_list = tx_list[-5:]

            lines: List[str] = []
            for tx in tx_list:
                tid = tx.get("transaction_id", "")
                amt = tx.get("amount", 0.0) or 0.0
                sb = tx.get("sender_bank", "Unknown")
                rb = tx.get("receiver_bank", "Unknown")
                ts = tx.get("timestamp", "")
                sc = tx.get("score", 0.0) or 0.0
                try:
                    amt_f = float(amt)
                except (TypeError, ValueError):
                    amt_f = 0.0
                try:
                    sc_f = float(sc)
                except (TypeError, ValueError):
                    sc_f = 0.0

                lines.append(f"ID: {tid} | €{amt_f:,.2f} | {sb} → {rb} | score={sc_f:.3f} | {ts}")

            tx_hover = "<br>".join(lines) if lines else "No transaction details"

            traces.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode="lines",
                    line=dict(width=width, color=color),
                    hoverinfo="text",
                    text=[tx_hover, tx_hover, ""],
                    customdata=[[sender, receiver, max_score], [sender, receiver, max_score], [sender, receiver, max_score]],
                    hovertemplate=(
                        "<b>Sender:</b> %{customdata[0]}<br>"
                        "<b>Receiver:</b> %{customdata[1]}<br>"
                        "<b>Edge max score:</b> %{customdata[2]:.3f}<br><br>"
                        "<b>Recent transactions:</b><br>%{text}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

        return traces

    def _create_node_trace(self, size_by: str = "transactions") -> go.Scatter:
        if not self.pos:
            self._compute_layout("flow", seed=42)

        try:
            thr = float(self.score_threshold)
        except (TypeError, ValueError):
            thr = 0.5

        # mark fraud-involved nodes
        fraud_nodes = set()
        for u, v, d in self.graph.edges(data=True):
            raw = d.get("max_score", 0.0)
            try:
                sc = float(raw) if raw not in (None, "") else 0.0
            except (TypeError, ValueError):
                sc = 0.0
            if sc >= thr:
                fraud_nodes.add(u)
                fraud_nodes.add(v)

        node_x: List[float] = []
        node_y: List[float] = []
        node_size: List[float] = []
        node_color: List[str] = []
        node_text: List[str] = []

        for node in self.graph.nodes():
            x, y = self.pos[node]
            node_x.append(x)
            node_y.append(y)

            in_edges = list(self.graph.in_edges(node, data=True))
            out_edges = list(self.graph.out_edges(node, data=True))

            in_deg = len(in_edges)
            out_deg = len(out_edges)
            deg = in_deg + out_deg

            if size_by == "degree":
                size = 10.0 + 8.0 * math.log1p(deg)
            else:
                total_tx = 0
                for _, _, d in in_edges + out_edges:
                    w = d.get("weight", 1)
                    try:
                        total_tx += int(w)
                    except (TypeError, ValueError):
                        total_tx += 1
                size = 10.0 + 7.0 * math.log1p(total_tx)

            size = max(10.0, min(size, 44.0))
            node_size.append(size)

            if node in fraud_nodes:
                node_color.append("rgba(255, 170, 80, 0.96)")  # highlight
            else:
                node_color.append("rgba(120, 180, 255, 0.92)")

            # simple hover summary
            total_in_amount = 0.0
            total_out_amount = 0.0
            fraud_edges_count = 0
            max_sc_node = 0.0

            for _, _, d in in_edges:
                val = d.get("total_amount", 0.0)
                try:
                    total_in_amount += float(val)
                except (TypeError, ValueError):
                    pass
                raw = d.get("max_score", 0.0)
                try:
                    sc = float(raw) if raw not in (None, "") else 0.0
                except (TypeError, ValueError):
                    sc = 0.0
                if sc >= thr:
                    fraud_edges_count += 1
                max_sc_node = max(max_sc_node, sc)

            for _, _, d in out_edges:
                val = d.get("total_amount", 0.0)
                try:
                    total_out_amount += float(val)
                except (TypeError, ValueError):
                    pass
                raw = d.get("max_score", 0.0)
                try:
                    sc = float(raw) if raw not in (None, "") else 0.0
                except (TypeError, ValueError):
                    sc = 0.0
                if sc >= thr:
                    fraud_edges_count += 1
                max_sc_node = max(max_sc_node, sc)

            node_text.append(
                f"<b>Account:</b> {node}<br>"
                f"<b>In-degree:</b> {in_deg} | <b>Out-degree:</b> {out_deg}<br>"
                f"<b>Total received:</b> €{total_in_amount:,.2f}<br>"
                f"<b>Total sent:</b> €{total_out_amount:,.2f}<br>"
                f"<b>Max edge score:</b> {max_sc_node:.3f}<br>"
                f"<b>Fraud edges (≥thr):</b> {fraud_edges_count}"
            )

        return go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            text=node_text,
            marker=dict(size=node_size, color=node_color, line=dict(width=1, color="white")),
            showlegend=False,
        )

    def create_plotly_figure(
        self,
        layout_algorithm: str = "flow",
        node_size_by: str = "transactions",
        seed: int = 42,
    ) -> go.Figure:
        self._compute_layout(layout_algorithm, seed=seed)

        fig = go.Figure()

        # Legend (professional)
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="lines",
                line=dict(width=4, color="rgba(200, 60, 60, 0.92)"),
                name="Fraud (edge max score ≥ threshold)",
                hoverinfo="skip",
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="lines",
                line=dict(width=4, color="rgba(140, 140, 140, 0.18)"),
                name="Normal",
                hoverinfo="skip",
                showlegend=True,
            )
        )

        for tr in self._create_edge_traces():
            fig.add_trace(tr)
        fig.add_trace(self._create_node_trace(size_by=node_size_by))

        fig.update_layout(
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor="white",
            plot_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        )
        return fig

    # -----------------------------
    # Metrics
    # -----------------------------
    def get_network_metrics(self) -> Dict[str, Any]:
        G = self.graph
        metrics: Dict[str, Any] = {}

        n_nodes = len(G)
        n_edges = len(G.edges)

        metrics["n_nodes"] = int(n_nodes)
        metrics["n_edges"] = int(n_edges)

        if n_nodes > 0:
            degree_values: List[int] = []
            hubs: List[Dict[str, Any]] = []

            for node in G.nodes:
                in_deg = len(list(G.in_edges(node)))
                out_deg = len(list(G.out_edges(node)))
                total_deg = in_deg + out_deg
                degree_values.append(total_deg)
                hubs.append({"account_id": node, "degree": total_deg, "in_degree": in_deg, "out_degree": out_deg})

            metrics["average_degree"] = float(sum(degree_values)) / float(len(degree_values)) if degree_values else 0.0

            try:
                metrics["density"] = float(nx.density(G))
            except Exception:
                metrics["density"] = 0.0

            try:
                metrics["n_connected_components"] = int(nx.number_weakly_connected_components(G))
            except Exception:
                metrics["n_connected_components"] = 1

            hubs.sort(key=lambda x: x["degree"], reverse=True)
            metrics["top_hubs"] = hubs[:10]
        else:
            metrics["average_degree"] = 0.0
            metrics["density"] = 0.0
            metrics["n_connected_components"] = 0
            metrics["top_hubs"] = []

        # fraud edges count
        try:
            thr = float(self.score_threshold)
        except (TypeError, ValueError):
            thr = 0.5

        fraud_edges = 0
        for _, _, d in G.edges(data=True):
            raw = d.get("max_score", 0.0)
            try:
                sc = float(raw) if raw not in (None, "") else 0.0
            except (TypeError, ValueError):
                sc = 0.0
            if sc >= thr:
                fraud_edges += 1
        metrics["num_fraud_edges"] = int(fraud_edges)

        total_amount = 0.0
        for _, _, d in G.edges(data=True):
            val = d.get("total_amount", 0.0)
            try:
                total_amount += float(val)
            except (TypeError, ValueError):
                pass
        metrics["total_amount"] = float(total_amount)

        metrics["score_threshold_used"] = float(thr)
        metrics["score_column_used"] = self.score_column_used

        return metrics


def create_transaction_network(
    df: pd.DataFrame,
    score_threshold: Optional[float] = None,
    display_transactions: int = 400,   # user-facing control
    layout_algorithm: str = "flow",    # "flow" | "spring" | "kamada_kawai"
    node_size_by: str = "transactions",
    seed: int = 42,
) -> Tuple[go.Figure, Dict[str, Any]]:
    # dataset totals
    total_tx = int(len(df)) if df is not None else 0

    try:
        thr = float(score_threshold) if score_threshold is not None else 0.5
    except (TypeError, ValueError):
        thr = 0.5

    # select professional subset
    selected_df, counts = select_transactions_for_network(
        df=df,
        score_threshold=thr,
        display_transactions=int(display_transactions),
        seed=seed,
    )

    # compute flagged in full dataset (for your "20k, 209 flagged" line)
    score_col = _guess_col(
        df,
        direct=["enhanced_aml_score", "anomaly_score", "aml_score", "score"],
        keywords=["aml", "anomaly", "score"],
    )
    if score_col and df is not None and score_col in df.columns:
        scores = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)
        flagged_tx = int((scores >= thr).sum())
    else:
        flagged_tx = 0

    builder = TransactionNetworkGraph()
    builder.build_graph(selected_df, score_threshold=thr)

    fig = builder.create_plotly_figure(layout_algorithm=layout_algorithm, node_size_by=node_size_by, seed=seed)
    metrics = builder.get_network_metrics()

    metrics.update(counts)
    metrics["total_transactions_in_df"] = total_tx
    metrics["flagged_transactions_in_df"] = flagged_tx

    return fig, metrics
