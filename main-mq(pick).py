import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

# ================== 核心函数定义 ==================
def calculate_30d_annualized(df, date_col, nav_col, window=30, smooth_window=7):
    """
    计算平滑处理的30天年化收益率（带日期列参数）
    参数：
    - date_col: 动态日期列名
    - smooth_window: 使用7天移动平均进行平滑
    """
    # 按日期排序确保计算正确（使用动态日期列名）
    df = df.sort_values(date_col)
    
    # 计算日收益率
    df['daily_return'] = df[nav_col].pct_change()
    
    # 计算滚动30天累计收益
    rolling_return = (1 + df['daily_return']).rolling(window=window, min_periods=1).apply(np.prod, raw=True) - 1
    
    # 年化处理（365天基准）
    annualized = (1 + rolling_return) ** (365/window) - 1
    
    # 使用移动平均平滑前段波动
    smoothed = annualized.rolling(smooth_window, min_periods=1, center=True).mean()
    
    return smoothed * 100  # 转换为百分比

def calculate_max_drawdown(data):
    """计算最大回撤（带类型转换和空值处理）"""
    nav_series = pd.to_numeric(data, errors='coerce')
    if nav_series.empty or nav_series.isna().all():
        return np.nan
    cumulative_max = nav_series.cummax()
    drawdown = (nav_series - cumulative_max) / cumulative_max
    return drawdown.min()

def preprocess_data(data, date_col, nav_col):
    """数据预处理管道（增强日期处理）"""
    # 防御性检查
    if date_col not in data.columns or nav_col not in data.columns:
        raise ValueError("指定的列不存在于数据中")
    
    # 统一解析日期格式
    def parse_date(date_str):
        """支持多种日期格式的解析"""
        try:
            # 尝试直接解析常见日期格式
            return pd.to_datetime(date_str, errors='coerce')
        except Exception:
            # 替换中文日期格式
            date_str = str(date_str).replace('年', '-').replace('月', '-').replace('日', '')
            return pd.to_datetime(date_str, errors='coerce')

    # 类型转换
    data[date_col] = data[date_col].apply(parse_date)
    data[nav_col] = pd.to_numeric(data[nav_col], errors='coerce')
    
    # 数据过滤
    processed_data = data.dropna(subset=[date_col, nav_col]).sort_values(date_col)
    
    # 有效性验证
    if processed_data.empty:
        raise ValueError("预处理后数据为空，请检查原始数据质量")
    
    return processed_data

# ================== Streamlit界面 ==================
st.title("\U0001F4C8 交互式净值与收益率分析")

@st.cache_data
def load_data(uploaded_file):
    """数据加载函数"""
    file_type = uploaded_file.name.split(".")[-1]
    if file_type == "csv":
        return pd.read_csv(uploaded_file)
    elif file_type == "xlsx":
        return pd.read_excel(uploaded_file)
    return None

# 文件上传组件
uploaded_file = st.file_uploader("请上传 CSV 或 Excel 文件", type=["csv", "xlsx"])

if uploaded_file is not None:
    # 数据加载
    raw_data = load_data(uploaded_file)
    if raw_data is None:
        st.error("不支持的文件格式！")
        st.stop()

    # 侧边栏配置
    st.sidebar.subheader("\U0001F527 分析配置")
    columns = raw_data.columns.tolist()
    
    # 列选择
    product_col = st.sidebar.selectbox("选择产品名称列", columns)
    date_col = st.sidebar.selectbox("选择日期列", columns)
    nav_col = st.sidebar.selectbox("选择累计单位净值列", columns)

    # 数据预处理（新增部分）
    try:
        data = preprocess_data(raw_data.copy(), date_col, nav_col)
    except Exception as e:
        st.error(f"数据预处理失败：{str(e)}")
        st.write("原始数据示例：", raw_data.head(2))
        st.stop()

    # 显示数据质量报告（新增）
    st.sidebar.markdown(f"""
    **\U0001F4CA 数据质量报告**
    - 原始记录数：{len(raw_data)}
    - 有效记录数：{len(data)} 
    - 时间范围：{data[date_col].min().strftime('%Y-%m-%d')} ~ {data[date_col].max().strftime('%Y-%m-%d')}
    """)


    # 时间范围设置
    st.sidebar.subheader("⏰ 时间范围")
    preset = st.sidebar.radio("快捷选择", 
        ["最近1年", "年初至今", "自定义"], 
        index=2,
        horizontal=True)

    # 数据中的最小和最大日期
    min_date = pd.to_datetime(data[date_col].min())
    max_date = pd.to_datetime(data[date_col].max())

    if preset == "最近1年":
        start_date = max_date - pd.DateOffset(years=1)
        end_date = max_date
    elif preset == "年初至今":
        # 基于 max_date 动态确定年份
        start_date = pd.Timestamp(year=max_date.year, month=1, day=1)
        end_date = max_date
    else:
        start_date, end_date = st.sidebar.date_input(
            "自定义时间范围",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )

    if start_date > end_date:
        st.sidebar.error("错误：开始时间不能晚于结束时间")
        st.stop()


    # 产品筛选（默认全选）
    all_products = data[product_col].unique()
    selected_products = st.sidebar.multiselect(
        "筛选展示产品",
        options=all_products,
        default=all_products
    )

    filtered_data = data[
        (data[date_col] >= pd.Timestamp(start_date)) & 
        (data[date_col] <= pd.Timestamp(end_date)) &
        (data[product_col].isin(selected_products))
    ]

    if filtered_data.empty:
        st.error("错误：筛选后无可用数据")
        st.stop()

    grouped = filtered_data.groupby(product_col)

    # 指标说明
    st.subheader("📊 收益风险分析表")
    st.markdown("""
    **指标说明**  
    🟢 年化收益率：基于实际持有天数计算的年化收益（公式：`(1+区间收益)^(365/天数)-1`）  
    🔴 最大回撤：选定时间段内从最高点到最低点的最大跌幅  
    📊 年化收益率：见下方年化收益率趋势图表  
    ⚠️ 数据不足：表示该时间段内没有足够历史数据
    """)

     # 计算逻辑
    end_date_actual = filtered_data[date_col].max()
    periods = {
        "近1个月": pd.DateOffset(months=1),
        "近3个月": pd.DateOffset(months=3),
        "近半年": pd.DateOffset(months=6),
        "近1年": pd.DateOffset(years=1),
        "自定义时间": None  # 自定义时间需要单独处理
    }

    # 修改后的完整计算逻辑
    results = []
    for name, group in grouped:
        group = group.set_index(date_col)
        
        try:
            latest_nav = group[nav_col].asof(end_date_actual)
            if pd.isna(latest_nav):
                continue
        except KeyError:
            continue

        # 最大回撤
        max_drawdown = calculate_max_drawdown(group[nav_col])
        
        # 初始化收益率字典
        period_returns = {}
        
        # 年化收益率计算
        for period_name, offset in periods.items():
            if period_name == "自定义时间":  # 自定义时间单独处理
                start_date_custom = pd.Timestamp(start_date)
                end_date_custom = pd.Timestamp(end_date)
                start_nav = group[nav_col].asof(start_date_custom)
                end_nav = group[nav_col].asof(end_date_custom)
                if pd.notna(start_nav) and pd.notna(end_nav):
                    days = (end_date_custom - start_date_custom).days
                    if days <= 0:
                        period_return = np.nan
                    else:
                        simple_return = (end_nav / start_nav) - 1
                        annualized_return = (1 + simple_return) ** (365 / days) - 1
                        period_return = annualized_return
                else:
                    period_return = np.nan
            else:  # 其他时间段的计算逻辑
                start_date_period = end_date_actual - offset
                try:
                    start_nav = group[nav_col].asof(start_date_period)
                    if pd.notna(start_nav):
                        days = (end_date_actual - start_date_period).days
                        if days <= 0:
                            period_return = np.nan
                        else:
                            simple_return = (latest_nav / start_nav) - 1
                            annualized_return = (1 + simple_return) ** (365 / days) - 1
                            period_return = annualized_return
                    else:
                        period_return = np.nan
                except KeyError:
                    period_return = np.nan
            
            # 存储计算结果
            period_returns[period_name] = period_return

        results.append({
            "产品名称": name,
            "最大回撤": max_drawdown,
            **period_returns
        })

    # 结果数据框
    results_df = pd.DataFrame(results)
    results_df["最大回撤"] = results_df["最大回撤"].apply(lambda x: f"{x:.2%}")
    for period in periods.keys():
        results_df[period] = results_df[period].apply(
            lambda x: f"{x:.2%}" if pd.notna(x) else "数据不足"
        )

    # 显示结果表格
    st.dataframe(
        results_df,
        use_container_width=True
    )

    # 净值曲线
    st.subheader("📈 净值走势")

    # 创建净值曲线图表
    fig_nav = px.line(
        filtered_data,
        x=date_col,
        y=nav_col,
        color=product_col,
        title='',
        labels={date_col: '日期', nav_col: '累计单位净值 (元)'},
        line_shape="spline",
        render_mode="svg"
    )

    # 更新线条宽度和悬停信息
    fig_nav.update_traces(
        line_width=1.5,
        hovertemplate=(
            "%{y:.4f}"       # 累计单位净值保留4位小数
        )
    )

    # 更新布局
    fig_nav.update_layout(
        hovermode="x unified",  # 悬停时统一展示日期
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hoverlabel=dict(
            namelength=-1  # 删除悬停提示框中的图例横线
        )
    )

    # 显示图表
    st.plotly_chart(fig_nav, use_container_width=True)

    # 年化收益率（单独图表）
    st.subheader("📊 年化收益率趋势")

    # 确保日期列为日期类型
    filtered_data[date_col] = pd.to_datetime(filtered_data[date_col], errors='coerce')

    # 初始化年化收益率列
    filtered_data['年化收益率'] = np.nan

    # 修改后的年化收益率计算逻辑（替换原有循环部分）
    for name, group in filtered_data.groupby(product_col):
        # 按日期排序
        group = group.set_index(date_col).sort_index()

        # 生成完整时间索引
        full_idx = pd.date_range(start=group.index.min(), end=group.index.max(), freq='D')
        group = group.reindex(full_idx)

        # 标记有效净值数据的日期
        valid_dates = group.dropna(subset=[nav_col]).index

        # 遍历有效日期计算年化收益率
        for current_date in valid_dates:
            # 动态寻找30天前的有效数据点
            lookback_date = current_date - pd.DateOffset(days=30)
            historical_data = group[nav_col].loc[:current_date].dropna()

            # 如果30天前的数据缺失，尝试寻找更近的有效数据点
            if lookback_date not in historical_data.index:
                closest_date = historical_data.index[historical_data.index <= lookback_date].max()
                if pd.isna(closest_date):
                    continue  # 如果找不到有效数据点，跳过该日期
                lookback_date = closest_date

            # 获取起始净值和结束净值
            start_nav = historical_data.loc[lookback_date]
            end_nav = historical_data.loc[current_date]

            # 计算实际间隔天数
            actual_days = (current_date - lookback_date).days
            if actual_days == 0:
                continue

            # 计算年化收益率
            cumulative_return = (end_nav / start_nav) - 1
            annualized_return = (1 + cumulative_return) ** (365 / actual_days) - 1

            # 更新到数据框
            filtered_data.loc[
                (filtered_data[product_col] == name) & 
                (filtered_data[date_col] == current_date),
                '年化收益率'
            ] = annualized_return * 100

    # 检查年化收益率列是否存在且为数值类型
    if '年化收益率' not in filtered_data.columns:
        raise ValueError("filtered_data 数据框中缺少 '年化收益率' 列，请检查计算逻辑。")
    if not pd.api.types.is_numeric_dtype(filtered_data['年化收益率']):
        raise ValueError("'年化收益率' 列不是数值类型，请检查计算逻辑。")

    # 处理空值
    filtered_data = filtered_data.dropna(subset=['年化收益率'])

    # 调试输出
    print("绘图数据：")
    print(filtered_data.head())

    # 绘制近一个月的年化收益率趋势图
    fig_annual = px.line(
        filtered_data,
        x=date_col,
        y='年化收益率',
        color=product_col,
        title='近一个月的年化收益率趋势',
        line_shape="spline",
        labels={date_col: '日期', '年化收益率': '年化收益率 (%)'},
        render_mode="svg"
    )

    # 更新图表样式，设置悬停信息
    fig_annual.update_traces(
        line_width=1.5,
        hovertemplate=(
            "%{y:.2f}%"       # 仅显示收益率百分比
        ),
    )
    # 更新布局配置（添加以下参数到年化收益率图表）
    fig_annual.update_layout(
        hovermode="x unified",  # 统一横向悬停
        hoverlabel=dict(
            namelength=-1  # 隐藏图例横线
        )
    )


    # 显示图表
    st.plotly_chart(fig_annual, use_container_width=True)


else:
    st.info("👋 请上传数据文件开始分析")
    st.markdown("""
    ### 使用说明
    1. 上传包含以下字段的金融产品数据：
       - 产品名称列
       - 日期列 (YYYY-MM-DD格式)
       - 累计净值列
    2. 在侧边栏完成列名匹配
    3. 选择分析时间范围和产品
    4. 查看交互式分析结果
    """)
