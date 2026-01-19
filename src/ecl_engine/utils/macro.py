def normalize_macro_z_columns(df):
    rename = {}
    # accept common aliases
    if "unemp_z" in df.columns and "unemployment_z" not in df.columns:
        rename["unemp_z"] = "unemployment_z"
    if "gdp_z" in df.columns and "gdp_yoy_z" not in df.columns:
        rename["gdp_z"] = "gdp_yoy_z"
    if "rate_z" in df.columns and "policy_rate_z" not in df.columns:
        rename["rate_z"] = "policy_rate_z"

    df = df.rename(columns=rename)

    required = ["unemployment_z", "gdp_yoy_z", "policy_rate_z"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Macro z columns missing: {missing}. Available: {list(df.columns)}")
    return df
