
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

st.set_page_config(
    page_title="MindTrace AI",
    page_icon="🧠",
    layout="wide",
)


@st.cache_data(show_spinner="Loading student dataset...")
def load_dataset():
    dataset = fetch_ucirepo(id=320)

    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    if "G3" in targets.columns:
        final_grade = targets["G3"]
    elif "G3" in features.columns:
        final_grade = features["G3"]
        features = features.drop(columns=["G3"])
    else:
        raise ValueError(
            "The dataset does not contain the expected G3 grade column."
        )

    df = features.copy()
    df["G3"] = pd.to_numeric(final_grade, errors="coerce")
    df = df.dropna(subset=["G3"]).copy()
    df["G3"] = df["G3"].astype(int)

    return df


@st.cache_resource(show_spinner="Training AI models...")
def train_models():
    df = load_dataset()

    feature_cols = [
        "age", "Medu", "Fedu", "traveltime", "studytime",
        "failures", "schoolsup", "famsup", "paid", "activities",
        "higher", "internet", "famrel", "freetime", "goout",
        "Dalc", "Walc", "health", "absences",
    ]

    missing = [
        col for col in feature_cols
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Dataset is missing columns: " + ", ".join(missing)
        )

    X = df[feature_cols].copy()
    y = (df["G3"] < 10).astype(int)

    if y.nunique() < 2:
        raise ValueError(
            "The dataset needs examples from both outcome classes."
        )

    cat_cols = X.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    num_cols = [
        col for col in X.columns
        if col not in cat_cols
    ]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)

    # Experimental threshold for identifying possible academic risk.
    threshold = 0.35

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1-score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "Confusion matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ),
        "Threshold": threshold,
    }

    # Unsupervised learning: discover three learning-pattern groups.
    cluster_cols = [
        "studytime", "failures", "absences",
        "freetime", "goout", "health",
    ]

    cluster_scaler = StandardScaler()
    cluster_values = cluster_scaler.fit_transform(
        df[cluster_cols].astype(float)
    )

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10,
    )

    df["Cluster"] = kmeans.fit_predict(cluster_values)

    return (
        df,
        feature_cols,
        model,
        metrics,
        cluster_cols,
        cluster_scaler,
        kmeans,
    )


# -------------------- APPLICATION HEADER --------------------

st.title("🧠 MindTrace AI")
st.subheader("Student Learning Pattern & Academic Risk Analyzer")

st.write(
    "An AI-powered educational project combining supervised and "
    "unsupervised machine learning to explore student learning patterns."
)

st.info(
    "This is an educational demonstration using a public dataset. "
    "Predictions are estimates, not guarantees, and should not be used "
    "to make high-stakes decisions about students."
)

# -------------------- LOAD AND TRAIN --------------------

try:
    (
        df,
        feature_cols,
        model,
        metrics,
        cluster_cols,
        cluster_scaler,
        kmeans,
    ) = train_models()

except Exception as exc:
    st.error(
        "MindTrace AI could not load the dataset or train its models."
    )
    st.write(
        "Check your internet connection, installed packages, "
        "and the error details below."
    )
    st.exception(exc)
    st.stop()

# -------------------- DASHBOARD TABS --------------------


tab1, tab2, tab3, tab4 = st.tabs([
    "🔎 Student Analyzer",
    "📊 Dataset Insights",
    "🤖 Model Evaluation",
    "🧪 What-if Simulator",
])



# ==================== TAB 1 ====================

with tab1:
    st.header("Analyze a Student Learning Profile")

    st.write(
        "Enter a sample profile and click Analyze to view the "
        "estimated risk and learning-pattern group."
    )

    with st.form("student_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider("Age", 15, 22, 17)

            studytime = st.select_slider(
                "Weekly study-time category",
                options=[1, 2, 3, 4],
                value=2,
                format_func=lambda value: {
                    1: "Under 2 hours",
                    2: "2–5 hours",
                    3: "5–10 hours",
                    4: "Over 10 hours",
                }[value],
            )

            failures = st.select_slider(
                "Past class failures",
                options=[0, 1, 2, 3, 4],
                value=0,
            )

            absences = st.slider(
                "Absences",
                min_value=0,
                max_value=30,
                value=3,
            )

            traveltime = st.select_slider(
                "Travel-time category",
                options=[1, 2, 3, 4],
                value=2,
            )

        with col2:
            Medu = st.select_slider(
                "Mother's education (0–4)",
                options=[0, 1, 2, 3, 4],
                value=2,
            )

            Fedu = st.select_slider(
                "Father's education (0–4)",
                options=[0, 1, 2, 3, 4],
                value=2,
            )

            famrel = st.slider(
                "Family relationship (1–5)", 1, 5, 4
            )

            freetime = st.slider(
                "Free time (1–5)", 1, 5, 3
            )

            goout = st.slider(
                "Going out (1–5)", 1, 5, 3
            )

        with col3:
            Dalc = st.slider(
                "Workday lifestyle indicator (1–5)",
                1, 5, 1,
            )

            Walc = st.slider(
                "Weekend lifestyle indicator (1–5)",
                1, 5, 2,
            )

            health = st.slider(
                "Self-reported health (1–5)",
                1, 5, 4,
            )

            schoolsup = st.selectbox(
                "Extra school support?", ["no", "yes"]
            )

            famsup = st.selectbox(
                "Family support?", ["yes", "no"]
            )

            paid = st.selectbox(
                "Paid extra classes?", ["no", "yes"]
            )

            activities = st.selectbox(
                "Extracurricular activities?", ["yes", "no"]
            )

            higher = st.selectbox(
                "Wants higher education?", ["yes", "no"]
            )

            internet = st.selectbox(
                "Internet access at home?", ["yes", "no"]
            )

        submitted = st.form_submit_button(
            "Analyze Learning Profile",
            type="primary",
        )

    if submitted:
        profile = pd.DataFrame([{
            "age": age,
            "Medu": Medu,
            "Fedu": Fedu,
            "traveltime": traveltime,
            "studytime": studytime,
            "failures": failures,
            "schoolsup": schoolsup,
            "famsup": famsup,
            "paid": paid,
            "activities": activities,
            "higher": higher,
            "internet": internet,
            "famrel": famrel,
            "freetime": freetime,
            "goout": goout,
            "Dalc": Dalc,
            "Walc": Walc,
            "health": health,
            "absences": absences,
        }])[feature_cols]

        risk_probability = float(
            model.predict_proba(profile)[0, 1]
        )

        predicted_risk = (
            risk_probability >= metrics["Threshold"]
        )

        cluster_input = pd.DataFrame([{
            "studytime": studytime,
            "failures": failures,
            "absences": absences,
            "freetime": freetime,
            "goout": goout,
            "health": health,
        }])[cluster_cols]

        cluster_id = int(
            kmeans.predict(
                cluster_scaler.transform(
                    cluster_input.astype(float)
                )
            )[0]
        )

        result1, result2, result3 = st.columns(3)

        result1.metric(
            "Estimated Risk Probability",
            f"{risk_probability:.0%}",
        )

        result2.metric(
            "Model Category",
            "Needs Support" if predicted_risk
            else "Lower Estimated Risk",
        )

        result3.metric(
            "Learning Pattern Group",
            f"Group {cluster_id + 1}",
        )

        if predicted_risk:
            st.warning(
                "The model flags this profile for possible academic "
                "support. This is an estimate, not a guaranteed outcome."
            )
        else:
            st.success(
                "The model does not flag this profile as higher risk. "
                "This does not guarantee future academic results."
            )

        
        st.subheader("Personalized Study Recommendations")

        suggestions = []
        priority = "Normal"

        if predicted_risk:
            priority = "High"
            suggestions.append({
                "title": "Create a focused study plan",
                "reason": (
                    "The model estimates a higher possibility of "
                    "falling below the selected pass threshold."
                ),
                "action": (
                    "Plan short daily study sessions and revise "
                    "difficult topics regularly."
                ),
            })

        if studytime <= 1:
            suggestions.append({
                "title": "Increase regular study time",
                "reason": "The selected study-time category is low.",
                "action": (
                    "Start with 25-minute focused sessions and "
                    "take short breaks between sessions."
                ),
            })

        if absences >= 5:
            suggestions.append({
                "title": "Catch up on missed lessons",
                "reason": "The entered absence count is relatively high.",
                "action": (
                    "Review missed topics and set aside time each "
                    "week to catch up."
                ),
            })

        if failures >= 1:
            suggestions.append({
                "title": "Focus on challenging subjects",
                "reason": "Previous class failures were reported.",
                "action": (
                    "Break difficult subjects into smaller goals "
                    "and ask a teacher for help when needed."
                ),
            })

        if famsup == "no":
            suggestions.append({
                "title": "Build a support system",
                "reason": "Family support was entered as unavailable.",
                "action": (
                    "Consider reaching out to a teacher, classmate, "
                    "mentor, or another trusted person for study support."
                ),
            })

        if not suggestions:
            suggestions.append({
                "title": "Maintain your learning routine",
                "reason": (
                    "The entered profile did not trigger the "
                    "specific rules above."
                ),
                "action": (
                    "Continue revising regularly and review your "
                    "learning progress every week."
                ),
            })

        st.write(
            "These suggestions are based on the entered profile "
            "and model estimate. They are guidance, not guarantees."
        )

        st.metric("Recommendation Priority", priority)

        for index, item in enumerate(suggestions, start=1):
            with st.expander(
                f"{index}. {item['title']}",
                expanded=True,
            ):
                st.write("**Why this is suggested:**")
                st.write(item["reason"])
                st.write("**What you can do:**")
                st.write(item["action"])

        st.caption(
            "Educational suggestions only. The model may make "
            "incorrect predictions, so review results with human "
            "judgement rather than treating them as facts."
        )



# ==================== TAB 2 ====================

with tab2:
    st.header("Student Dataset Insights")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("Student Records", len(df))
    metric2.metric("Input Features", len(feature_cols))
    metric3.metric("Discovered Groups", 3)

    chart_df = df.copy()

    chart_df["Outcome"] = np.where(
        chart_df["G3"] < 10,
        "Below pass threshold",
        "At/above pass threshold",
    )

    fig1 = px.histogram(
        chart_df,
        x="studytime",
        color="Outcome",
        barmode="group",
        title="Grade Outcome by Study-Time Category",
        labels={
            "studytime": "Study-Time Category",
            "count": "Number of Students",
        },
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Average Characteristics by Group")

    cluster_summary = (
        df.groupby("Cluster")[cluster_cols + ["G3"]]
        .mean()
        .round(2)
        .reset_index()
    )

    cluster_summary["Cluster"] = (
        cluster_summary["Cluster"].astype(str)
        .radd("Group ")
    )

    st.dataframe(
        cluster_summary,
        use_container_width=True,
        hide_index=True,
    )

    fig2 = px.scatter(
        df,
        x="studytime",
        y="absences",
        color=df["Cluster"].astype(str),
        size="G3",
        hover_data=["failures", "G3"],
        title="Learning Groups: Study Time vs Absences",
        labels={
            "studytime": "Study-Time Category",
            "absences": "Absences",
            "color": "Learning Group",
        },
    )

    st.plotly_chart(fig2, use_container_width=True)


# ==================== TAB 3 ====================

with tab3:
    st.header("Random Forest Model Evaluation")

    st.caption(
        f"Metrics use a held-out test split and an experimental "
        f"probability threshold of {metrics['Threshold']:.2f}. "
        "Results may differ on other student populations."
    )

    metric_columns = st.columns(4)

    for column, name in zip(
        metric_columns,
        ["Accuracy", "Precision", "Recall", "F1-score"],
    ):
        column.metric(name, f"{metrics[name]:.1%}")

    st.subheader("Confusion Matrix")

    cm_df = pd.DataFrame(
        metrics["Confusion matrix"],
        index=["Actual Lower Risk", "Actual At-Risk"],
        columns=["Predicted Lower Risk", "Predicted At-Risk"],
    )

    st.dataframe(
        cm_df,
        use_container_width=True,
    )

    st.subheader("Most Important Model Features")

    classifier = model.named_steps["classifier"]

    feature_names = (
        model.named_steps["preprocess"]
        .get_feature_names_out()
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": classifier.feature_importances_,
    }).sort_values(
        "Importance",
        ascending=False,
    ).head(12)

    fig3 = px.bar(
        importance_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Features Used by Random Forest",
    )

    st.plotly_chart(fig3, use_container_width=True)


# ==================== TAB 4: WHAT-IF SIMULATOR ====================

with tab4:
    st.header("🧪 What-if Simulator")

    st.write(
        "Compare two sample student profiles to see how changing "
        "study time, absences, or previous failures affects the "
        "model's estimated academic risk."
    )

    st.info(
        "This simulator uses a sample profile. It does not automatically "
        "reuse the profile entered in the Student Analyzer."
    )

    with st.form("what_if_form"):
        left, right = st.columns(2)

        with left:
            st.subheader("Current Profile")

            current_studytime = st.select_slider(
                "Weekly study time",
                options=[1, 2, 3, 4],
                value=1,
                format_func=lambda value: {
                    1: "Under 2 hours",
                    2: "2–5 hours",
                    3: "5–10 hours",
                    4: "Over 10 hours",
                }[value],
                key="current_studytime",
            )

            current_absences = st.slider(
                "Absences",
                min_value=0,
                max_value=30,
                value=12,
                key="current_absences",
            )

            current_failures = st.select_slider(
                "Past class failures",
                options=[0, 1, 2, 3, 4],
                value=2,
                key="current_failures",
            )

        with right:
            st.subheader("Modified Profile")

            new_studytime = st.select_slider(
                "Weekly study time ",
                options=[1, 2, 3, 4],
                value=3,
                format_func=lambda value: {
                    1: "Under 2 hours",
                    2: "2–5 hours",
                    3: "5–10 hours",
                    4: "Over 10 hours",
                }[value],
                key="new_studytime",
            )

            new_absences = st.slider(
                "Absences ",
                min_value=0,
                max_value=30,
                value=3,
                key="new_absences",
            )

            new_failures = st.select_slider(
                "Past class failures ",
                options=[0, 1, 2, 3, 4],
                value=0,
                key="new_failures",
            )

        compare = st.form_submit_button(
            "Compare Profiles",
            type="primary",
        )

    if compare:
        # The other features stay the same in both sample profiles.
        common_profile = {
            "age": 17,
            "Medu": 2,
            "Fedu": 2,
            "traveltime": 2,
            "schoolsup": "no",
            "famsup": "yes",
            "paid": "no",
            "activities": "yes",
            "higher": "yes",
            "internet": "yes",
            "famrel": 4,
            "freetime": 3,
            "goout": 3,
            "Dalc": 1,
            "Walc": 2,
            "health": 4,
        }

        current_profile = {
            **common_profile,
            "studytime": current_studytime,
            "absences": current_absences,
            "failures": current_failures,
        }

        modified_profile = {
            **common_profile,
            "studytime": new_studytime,
            "absences": new_absences,
            "failures": new_failures,
        }

        current_df = pd.DataFrame(
            [current_profile]
        )[feature_cols]

        modified_df = pd.DataFrame(
            [modified_profile]
        )[feature_cols]

        current_risk = float(
            model.predict_proba(current_df)[0, 1]
        )

        modified_risk = float(
            model.predict_proba(modified_df)[0, 1]
        )

        difference = (modified_risk - current_risk) * 100

        st.subheader("Comparison Results")

        result1, result2, result3 = st.columns(3)

        result1.metric(
            "Current Estimated Risk",
            f"{current_risk:.1%}",
        )

        result2.metric(
            "Modified Estimated Risk",
            f"{modified_risk:.1%}",
        )

        result3.metric(
            "Change in Risk Estimate",
            f"{difference:+.1f} percentage points",
            delta=f"{difference:+.1f} pp",
            delta_color="inverse",
        )

        comparison_df = pd.DataFrame({
            "Profile": ["Current", "Modified"],
            "Estimated Risk (%)": [
                current_risk * 100,
                modified_risk * 100,
            ],
        })

        fig_compare = px.bar(
            comparison_df,
            x="Profile",
            y="Estimated Risk (%)",
            color="Profile",
            text="Estimated Risk (%)",
            title="Estimated Academic Risk Comparison",
            range_y=[0, 100],
        )

        fig_compare.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
        )

        st.plotly_chart(
            fig_compare,
            use_container_width=True,
        )

        if difference < -0.05:
            st.success(
                "The modified profile has a lower estimated risk "
                "according to this model."
            )
        elif difference > 0.05:
            st.warning(
                "The modified profile has a higher estimated risk "
                "according to this model."
            )
        else:
            st.info(
                "The model's estimated risk is nearly unchanged "
                "for these two profiles."
            )

        st.caption(
            "This is a model-based comparison, not proof that changing "
            "these factors will cause a change in academic results. "
            "Other profile features are held constant. Predictions "
            "can be incorrect and should not be treated as guarantees."
        )

# ==================== SIDEBAR ====================

with st.sidebar:
    st.header("About MindTrace AI")

    st.write("**Supervised Learning:** Random Forest")
    st.write("**Unsupervised Learning:** K-Means")
    st.write("**Dashboard:** Streamlit")
    st.write("**Visualizations:** Plotly")
    st.write("**Dataset:** UCI Student Performance")

    st.markdown(
        "[Open the UCI dataset source]"
        "(https://archive.ics.uci.edu/dataset/320/student+performance)"
    )

    st.divider()

    st.caption(
        "The dataset contains Portuguese secondary-school students. "
        "Its results may not generalize to Indian college students. "
        "Use this project for educational demonstration only. "
        "Do not enter personally identifiable student data."
    )
