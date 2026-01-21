import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from logger import logger
from config import settings

class SmartProfiler:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
        # Automatic sampling for large datasets for expensive operations
        if len(df) > settings.LARGE_DATASET_THRESHOLD:
            logger.info(f"Large dataset detected ({len(df)} rows). Sampling {settings.SAMPLE_SIZE_LARGE} rows for analysis.")
            self.sample_df = df.sample(settings.SAMPLE_SIZE_LARGE, random_state=42)
        else:
            self.sample_df = df
        
        self.numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()

    def get_dataset_summary(self):
        """
        Basic dataset stats.
        """
        memory_usage = self.df.memory_usage(deep=True).sum() / (1024 * 1024) # MB
        return {
            "rows": int(self.df.shape[0]),
            "columns": int(self.df.shape[1]),
            "memory_mb": round(float(memory_usage), 2),
            "duplicates": int(self.df.duplicated().sum()),
            "numerical_columns": int(len(self.numerical_cols)),
            "categorical_columns": int(len(self.categorical_cols))
        }

    def run_diagnostic_tests(self):
        """
        Run a battery of diagnostic tests: Missingness, Normality, ANOVA, Chi-Square, Outliers, Cardinality.
        """
        results = []

        # 1. Missing Value Analysis
        for col in self.df.columns:
            missing = self.df[col].isnull().sum()
            if missing > 0:
                pct = (missing / len(self.df)) * 100
                results.append({
                    "Test": "Missing Value Analysis",
                    "Variable": col,
                    "Result": f"{missing} ({pct:.1f}%) missing",
                    "Interpretation": "High missingness" if pct > 50 else "Moderate/Low"
                })

        # 2. Normality Test - on numerical cols
        for col in self.numerical_cols:
            data = self.sample_df[col].dropna()
            if len(data) > 20 and data.std() > 0:
                stat, p = stats.normaltest(data)
                results.append({
                    "Test": "Normality",
                    "Variable": col,
                    "Result": f"p = {p:.2e}",
                    "Interpretation": "Normal" if p > 0.05 else "Non-Normal"
                })

        # 3. ANOVA (Numerical Target vs Categorical Feature)
        target = 'Purchase' if 'Purchase' in self.numerical_cols else (self.numerical_cols[0] if self.numerical_cols else None)
        
        if target:
            for col in self.categorical_cols:
                data = self.sample_df[[target, col]].dropna()
                groups = [group[target].values for name, group in data.groupby(col)]
                if len(groups) > 1:
                    stat, p = stats.f_oneway(*groups)
                    results.append({
                        "Test": "ANOVA",
                        "Variable": f"{target} vs {col}",
                        "Result": f"p = {p:.2e}",
                        "Interpretation": "Significant Diff" if p < 0.05 else "Not Significant"
                    })

        # 4. Chi-Square (Categorical vs Categorical)
        if len(self.categorical_cols) >= 2:
            c1, c2 = self.categorical_cols[0], self.categorical_cols[1]
            contingency = pd.crosstab(self.sample_df[c1], self.sample_df[c2])
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            results.append({
                "Test": "Chi-Square Independence",
                "Variable": f"{c1} vs {c2}",
                "Result": f"p = {p:.2e}",
                "Interpretation": "Dependent" if p < 0.05 else "Independent"
            })

        # 5. Outlier Detection (Z-Score)
        if target:
            data = self.df[target].dropna()
            z_scores = np.abs(stats.zscore(data))
            outliers = np.sum(z_scores > 3)
            results.append({
                "Test": "Outlier Detection (Z>3)",
                "Variable": target,
                "Result": f"{outliers}",
                "Interpretation": "Has Outliers" if outliers > 0 else "Clean"
            })

        # 6. Cardinality Check
        for col in self.df.columns:
            unique_count = self.df[col].nunique()
            if unique_count > 50 and col in self.categorical_cols: # Only flag high cardinality cats
                 results.append({
                    "Test": "Cardinality Check",
                    "Variable": col,
                    "Result": f"{unique_count}",
                    "Interpretation": "High Cardinality"
                })

        return pd.DataFrame(results)

    def get_skewness_kurtosis(self):
        """
        Calculate Skewness and Kurtosis for numerical columns.
        """
        metrics = []
        for col in self.numerical_cols:
            data = self.df[col].dropna()
            skew = stats.skew(data)
            kurt = stats.kurtosis(data)
            
            # Simple interpretation logic
            skew_interp = "Symmetric"
            if skew > 1: skew_interp = "Highly Right-Skewed"
            elif skew > 0.5: skew_interp = "Right-Skewed"
            elif skew < -1: skew_interp = "Highly Left-Skewed"
            elif skew < -0.5: skew_interp = "Left-Skewed"
            
            kurt_interp = "Normal"
            if kurt > 1: kurt_interp = "Leptokurtic"
            elif kurt < -1: kurt_interp = "Platykurtic"
            
            metrics.append({
                "Column": col,
                "Skewness": round(float(skew), 3),
                "Kurtosis": round(float(kurt), 3),
                "Interpretation": f"{kurt_interp} ({skew_interp})"
            })
            
        return pd.DataFrame(metrics)

    def get_correlation_matrix_str(self):
        """
        Get the correlation matrix of numerical columns as a string.
        """
        if len(self.numerical_cols) < 2:
            return "Not enough numerical columns for correlation."
        
        corr_matrix = self.df[self.numerical_cols].corr()
        return corr_matrix.to_string()

    # --- New Methods for Automated AI Reporting ---

    def get_descriptive_stats(self):
        """
        Get detailed descriptive statistics for numerical columns.
        """
        return self.df[self.numerical_cols].describe().to_string()

    def get_distribution_analysis(self):
        """
        Return skewness and kurtosis as a string for context.
        """
        metrics = self.get_skewness_kurtosis()
        if metrics.empty:
            return "No numerical columns."
        return metrics.to_string(index=False)

    def get_correlations(self):
        """
        Alias for get_correlation_matrix_str.
        """
        return self.get_correlation_matrix_str()

    def get_categorical_analysis(self):
        """
        Summarize top categories for categorical features.
        """
        summary = ""
        for col in self.categorical_cols:
            top = self.df[col].value_counts(normalize=True).head(5)
            summary += f"\nColumn: {col}\n{top.to_string()}\n"
        return summary

    def get_hierarchical_analysis(self):
        """
        Performs structural diagnostics to verify Parent-Child Lattice relationships 
        across the entire dataset.
        """
        results = {}
        
        # 1. Nesting Dependency (Strict Presence Hierarchy)
        # Testing if Column B presence implies Column A presence
        hierarchy_pairs = [
            ('Product_Category_1', 'Product_Category_2'),
            ('Product_Category_2', 'Product_Category_3')
        ]
        
        nesting_stats = {}
        for parent, child in hierarchy_pairs:
            if parent in self.df.columns and child in self.df.columns:
                child_exists = self.df[self.df[child].notnull()]
                parent_null_when_child_exists = child_exists[child_exists[parent].isnull()]
                nesting_stats[f"{parent}->{child}"] = {
                    "valid_nesting": len(parent_null_when_child_exists) == 0,
                    "violations": len(parent_null_when_child_exists)
                }
        results["nesting_logic"] = nesting_stats

        # 2. Shared Attribute (Lattice) Check
        # Does a child ID appear under multiple parents?
        lattice_check = {}
        if 'Product_Category_2' in self.df.columns and 'Product_Category_1' in self.df.columns:
            pc2_parents = self.df.groupby('Product_Category_2')['Product_Category_1'].nunique()
            shared_count = len(pc2_parents[pc2_parents > 1])
            lattice_check["pc2_is_shared_lattice"] = shared_count > 0
            lattice_check["pc2_shared_categories_count"] = shared_count
        results["lattice_logic"] = lattice_check

        # 3. Terminal Node Analysis (Sparsity Profiles)
        # Identifying if certain Parent values 'forbid' children
        if 'Product_Category_1' in self.df.columns and 'Product_Category_2' in self.df.columns:
            terminal_profiles = self.df.groupby('Product_Category_1').apply(
                lambda x: x['Product_Category_2'].notnull().mean()
            ).to_dict()
            results["terminal_node_profiles"] = terminal_profiles

        return results

    def generate_advanced_stats(self):
        """
        Generate advanced statistics to answer complex user queries.
        """
        report = "--- ADVANCED DATASET STATISTICS ---\n"
        
        # 1. Top 1% Spenders Profile
        if 'Purchase' in self.df.columns:
            threshold = self.df['Purchase'].quantile(0.99)
            top_1_df = self.df[self.df['Purchase'] >= threshold]
            rest_df = self.df[self.df['Purchase'] < threshold]
            
            report += f"\n1. TOP 1% SPENDERS (Purchase >= {threshold:.2f}):\n"
            for col in ['Gender', 'Age', 'City_Category', 'Occupation']:
                if col in self.df.columns:
                    top_dist = top_1_df[col].value_counts(normalize=True).head(1)
                    rest_dist = rest_df[col].value_counts(normalize=True).head(1)
                    report += f"   - Top {col}: {top_dist.index[0]} ({top_dist.values[0]:.1%}) vs Avg: {rest_dist.index[0]} ({rest_dist.values[0]:.1%})\n"

        # 2. Top 100 Users Analysis
        if 'User_ID' in self.df.columns and 'Purchase' in self.df.columns:
            user_spend = self.df.groupby('User_ID')['Purchase'].sum().sort_values(ascending=False)
            top_100_users = user_spend.head(100).index
            top_100_df = self.df[self.df['User_ID'].isin(top_100_users)]
            
            report += "\n2. TOP 100 HIGH-VALUE USERS:\n"
            for col in ['Occupation', 'City_Category']:
                if col in self.df.columns:
                    top_val = top_100_df[col].value_counts(normalize=True).head(1)
                    report += f"   - Most Common {col}: {top_val.index[0]} ({top_val.values[0]:.1%})\n"

        # 3. Product Strategy (Loss Leader vs Margin Driver)
        if 'Product_ID' in self.df.columns and 'Purchase' in self.df.columns:
            prod_stats = self.df.groupby('Product_ID')['Purchase'].agg(['count', 'mean']).sort_values('count', ascending=False)
            loss_leader = prod_stats.head(1)
            margin_driver = prod_stats.sort_values('mean', ascending=False).head(1)
            
            report += "\n3. PRODUCT STRATEGY:\n"
            report += f"   - Potential Loss Leader (High Vol): {loss_leader.index[0]} (Count: {loss_leader['count'].values[0]}, Avg Price: {loss_leader['mean'].values[0]:.2f})\n"
            report += f"   - Potential Margin Driver (High Price): {margin_driver.index[0]} (Avg Price: {margin_driver['mean'].values[0]:.2f}, Count: {margin_driver['count'].values[0]})\n"

        # 4. Stay Duration Impact
        if 'Stay_In_Current_City_Years' in self.df.columns and 'Purchase' in self.df.columns:
            stay_stats = self.df.groupby('Stay_In_Current_City_Years')['Purchase'].agg(['sum', 'mean'])
            report += "\n4. STAY DURATION IMPACT:\n"
            report += stay_stats.to_string() + "\n"

        # 5. Purchase Variance by Category
        if 'Product_Category_1' in self.df.columns and 'Purchase' in self.df.columns:
            var_stats = self.df.groupby('Product_Category_1')['Purchase'].var().sort_values(ascending=False).head(3)
            report += "\n5. HIGHEST PRICE VARIANCE (Product_Category_1):\n"
            report += var_stats.to_string() + "\n"

        return report

    def get_plot_bytes(self, fig):
        """
        Convert a Plotly figure to PNG bytes.
        """
        import plotly.io as pio
        return pio.to_image(fig, format="png")

    def generate_all_plots(self):
        """
        Generate specific visualizations: Distribution, Gender, Age, City, Heatmap, Pairwise.
        """
        import plotly.figure_factory as ff
        plots = {}
        
        # 1. Purchase Distribution (Histogram + KDE line)
        if 'Purchase' in self.numerical_cols:
            # distplot with KDE and Box plot on the side (using hist_data which creates density)
            # Note: ff.create_distplot doesn't support 'marginal="box"' directly like px.histogram.
            # It creates a separate rug plot by default. 
            # To get KDE + Box, we need to use px.histogram with marginal="box" and add a line trace for KDE, 
            # OR use ff.create_distplot and add a box trace manually.
            # The user asked to "Restore the box plot... with kde".
            # px.histogram can do marginal="box", but KDE line is tricky (only density).
            # ff.create_distplot does KDE line beautifully.
            # Let's try combining them by creating distplot and adding a box trace.
            
            import plotly.graph_objects as go
            
            hist_data = [self.df['Purchase'].dropna().values]
            group_labels = ['Purchase']

            # Create distplot (Histogram + KDE)
            fig_dist = ff.create_distplot(hist_data, group_labels, show_hist=True, show_rug=False)
            
            # Add Box plot trace above
            fig_dist.add_trace(go.Box(x=self.df['Purchase'], name='Purchase Box', xaxis='x', yaxis='y2', boxpoints=False))
            
            # Update layout to support the marginal box style layout
            fig_dist.update_layout(
                title="Purchase Distribution with KDE & Box", 
                template="plotly_white",
                yaxis2=dict(domain=[0.85, 1], showticklabels=False), # Box plot area
                yaxis=dict(domain=[0, 0.80]) # Histogram/KDE area
            )
            plots["Purchase Distribution"] = fig_dist

        # 2. Purchase by Gender
        if 'Gender' in self.categorical_cols and 'Purchase' in self.numerical_cols:
            # Calculate mean purchase by gender
            df_gender = self.df.groupby('Gender')['Purchase'].mean().reset_index()
            fig_gender = px.bar(df_gender, x='Gender', y='Purchase', 
                                title="Average Purchase by Gender", template="plotly_white")
            plots["Purchase by Gender"] = fig_gender
            
        # 3. Purchase by Age
        if 'Age' in self.categorical_cols and 'Purchase' in self.numerical_cols:
            df_age = self.df.groupby('Age')['Purchase'].mean().reset_index()
            # Sort age groups usually helpful, but they might be strings "0-17", "55+"
            # Simple alphanumeric sort usually works for these specific bins
            df_age = df_age.sort_values('Age') 
            fig_age = px.bar(df_age, x='Age', y='Purchase', 
                             title="Average Purchase by Age Group", template="plotly_white")
            plots["Purchase by Age"] = fig_age

        # 4. City Transaction Counts
        if 'City_Category' in self.categorical_cols:
            df_city = self.df['City_Category'].value_counts().reset_index()
            df_city.columns = ['City_Category', 'Count']
            fig_city = px.bar(df_city, x='City_Category', y='Count', 
                              title="Transaction Volume by City Category", template="plotly_white")
            plots["City Transactions"] = fig_city

        # 5. Gender vs Marital Status (Grouped Bar)
        if 'Gender' in self.categorical_cols and 'Marital_Status' in self.numerical_cols and 'Purchase' in self.numerical_cols:
            # Marital Status often int (0/1), convert to str for grouping
            temp_df = self.df.copy()
            temp_df['Marital_Status'] = temp_df['Marital_Status'].astype(str)
            df_grouped = temp_df.groupby(['Marital_Status', 'Gender'])['Purchase'].mean().reset_index()
            
            fig_grouped = px.bar(df_grouped, x='Marital_Status', y='Purchase', color='Gender', barmode='group',
                                 title="Purchase by Marital Status & Gender", template="plotly_white")
            plots["Gender vs Marital Status"] = fig_grouped
            
        # 6. Heatmap (Occupation vs Product_Category_1) - if they exist
        if 'Occupation' in self.numerical_cols and 'Product_Category_1' in self.numerical_cols:
             ct = pd.crosstab(self.df['Occupation'], self.df['Product_Category_1'])
             fig_heatmap = px.imshow(ct, title="Heatmap: Occupation vs Product Category", 
                                     template="plotly_white", aspect="auto")
             plots["Occupation vs Product Heatmap"] = fig_heatmap

        # 7. Pairwise Plot (Scatter Matrix) - Sample of 10000
        # Select important numerical columns to avoid a massive matrix
        cols_to_plot = ['Purchase', 'Occupation', 'Marital_Status', 'Product_Category_1', 'Product_Category_2']
        cols_available = [c for c in cols_to_plot if c in self.numerical_cols]
        
        if len(cols_available) > 1:
            sample_size = min(10000, len(self.df))
            pair_df = self.df[cols_available].sample(sample_size, random_state=42)
            fig_pair = px.scatter_matrix(pair_df, dimensions=cols_available, 
                                         title=f"Pairwise Scatter Matrix (Sample n={sample_size})",
                                         template="plotly_white",
                                         opacity=0.3,
                                         height=1000) # Increased height significantly
            # Reduce label size and adjust layout for matrix
            fig_pair.update_traces(diagonal_visible=False, showupperhalf=False)
            fig_pair.update_layout(
                font=dict(size=10),
                dragmode='select'
            )
            # Ensure tick labels are smaller if possible, though scatter_matrix uses axes labels mostly
            plots["Pairwise Plot"] = fig_pair

        return plots
