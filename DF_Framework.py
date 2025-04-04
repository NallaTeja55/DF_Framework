# Databricks notebook source
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from typing import Dict, List, Callable, Any, Union, Optional
import logging
import json
import inspect

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("transform_framework")

class DynamicTransformer:
    """
    A framework for dynamic PySpark DataFrame transformations using
    a configuration-driven approach.
    """
    
    def __init__(self, spark_session=None):
        """
        Initialize the transformer with optional SparkSession.
        
        Args:
            spark_session: Active SparkSession (optional)
        """
        self.spark = spark_session
        self._setup_transformation_registry()
        self._setup_action_registry()
        
    def _setup_transformation_registry(self):
        """Register all standard DataFrame transformations."""
        self.TRANSFORMATIONS = {
            # Basic transformations
            "select": self._transform_select,
            "filter": self._transform_filter,
            "where": self._transform_filter,  # Alias for filter
            "drop": self._transform_drop,
            "withColumn": self._transform_with_column,
            "withColumnRenamed": self._transform_rename_column,
            "selectExpr": self._transform_select_expr,
            
            # Aggregations
            "groupBy": self._transform_group_by,
            "agg": self._transform_aggregate,
            
            # Joins
            "join": self._transform_join,
            "unionByName": self._transform_union_by_name,
            
            # Window functions
            "withWindow": self._transform_with_window,
            
            # Advanced transformations
            "transform": self._transform_custom,
            "repartition": self._transform_repartition,
            "coalesce": self._transform_coalesce,
            "orderBy": self._transform_order_by,
            "limit": self._transform_limit,
            "sample": self._transform_sample,
            "distinct": self._transform_distinct,
            "cache": self._transform_cache,
            "unpersist": self._transform_unpersist,
            "checkpoint": self._transform_checkpoint,
            "dropDuplicates": self._transform_drop_duplicates,
            "fillna": self._transform_fill_na,
            "dropna": self._transform_drop_na,
            "explode": self._transform_explode,
            "pivot": self._transform_pivot,
        }
    
    def _setup_action_registry(self):
        """Register all DataFrame actions."""
        self.ACTIONS = {
            "collect": self._action_collect,
            "count": self._action_count,
            "show": self._action_show,
            "toPandas": self._action_to_pandas,
            "write": self._action_write,
            "explain": self._action_explain,
            "describe": self._action_describe,
            "first": self._action_first,
            "take": self._action_take,
            "head": self._action_head,
            "saveAsTable": self._action_save_as_table,
        }
    
    def register_custom_transformation(self, name: str, func: Callable):
        """
        Register a custom transformation function.
        
        Args:
            name: Name to register the transformation under
            func: Function that takes a DataFrame and params and returns a DataFrame
        """
        if name in self.TRANSFORMATIONS:
            logger.warning(f"Overriding existing transformation '{name}'")
        self.TRANSFORMATIONS[name] = func
        logger.info(f"Registered custom transformation: {name}")
    
    def register_custom_action(self, name: str, func: Callable):
        """
        Register a custom action function.
        
        Args:
            name: Name to register the action under
            func: Function that takes a DataFrame and params and returns a result
        """
        if name in self.ACTIONS:
            logger.warning(f"Overriding existing action '{name}'")
        self.ACTIONS[name] = func
        logger.info(f"Registered custom action: {name}")
    
    # ---- Core Framework Methods ----
    
    def apply_transformations(self, df: DataFrame, 
                             transformations: List[Dict[str, Any]]) -> DataFrame:
        """
        Apply a series of transformations to a DataFrame.
        
        Args:
            df: Input DataFrame
            transformations: List of transformation configurations
            
        Returns:
            Transformed DataFrame
        """
        result_df = df
        
        for i, transform_config in enumerate(transformations):
            try:
                operation = transform_config.get("operation")
                params = transform_config.get("params", {})
                
                if operation not in self.TRANSFORMATIONS:
                    raise ValueError(f"Unknown transformation: {operation}")
                
                logger.info(f"Applying transformation {i+1}/{len(transformations)}: {operation}")
                result_df = self.TRANSFORMATIONS[operation](result_df, params)
                
                # Optional caching if specified
                if transform_config.get("cache", False):
                    result_df = result_df.cache()
                    logger.info(f"Cached DataFrame after {operation}")
                
            except Exception as e:
                logger.error(f"Error applying transformation {operation}: {str(e)}")
                raise
                
        return result_df
    
    def execute_action(self, df: DataFrame, action_config: Dict[str, Any]) -> Any:
        """
        Execute an action on a DataFrame.
        
        Args:
            df: Input DataFrame
            action_config: Action configuration
            
        Returns:
            Result of the action
        """
        operation = action_config.get("operation")
        params = action_config.get("params", {})
        
        if operation not in self.ACTIONS:
            raise ValueError(f"Unknown action: {operation}")
        
        logger.info(f"Executing action: {operation}")
        return self.ACTIONS[operation](df, params)
    
    def transform_from_config(self, df: DataFrame, config: Dict[str, Any]) -> Any:
        """
        Transform a DataFrame using a complete configuration.
        
        Args:
            df: Input DataFrame
            config: Configuration with transformations and optional action
            
        Returns:
            Transformed DataFrame or action result
        """
        transformations = config.get("transformations", [])
        result_df = self.apply_transformations(df, transformations)
        
        # Execute action if specified
        action = config.get("action")
        if action:
            return self.execute_action(result_df, action)
        
        return result_df
    
    def transform_from_json(self, df: DataFrame, json_str: str) -> Any:
        """
        Transform a DataFrame using a JSON configuration string.
        
        Args:
            df: Input DataFrame
            json_str: JSON configuration
            
        Returns:
            Transformed DataFrame or action result
        """
        config = json.loads(json_str)
        return self.transform_from_config(df, config)
    
    def transform_from_file(self, df: DataFrame, file_path: str) -> Any:
        """
        Transform a DataFrame using a JSON configuration file.
        
        Args:
            df: Input DataFrame
            file_path: Path to JSON configuration file
            
        Returns:
            Transformed DataFrame or action result
        """
        with open(file_path, 'r') as f:
            config = json.load(f)
        return self.transform_from_config(df, config)
    
    # ---- Transformation Implementations ----
    
    def _transform_select(self, df: DataFrame, params: Dict) -> DataFrame:
        """Select specified columns."""
        cols = params.get("columns", [])
        return df.select(*cols)
    
    def _transform_filter(self, df: DataFrame, params: Dict) -> DataFrame:
        """Filter rows using a condition."""
        if "condition" in params:
            # String condition
            return df.filter(params["condition"])
        elif "column" in params and "value" in params:
            # Simple column=value filter
            column = params["column"]
            value = params["value"]
            operator = params.get("operator", "==")
            
            if operator == "==":
                return df.filter(F.col(column) == value)
            elif operator == "!=":
                return df.filter(F.col(column) != value)
            elif operator == ">":
                return df.filter(F.col(column) > value)
            elif operator == ">=":
                return df.filter(F.col(column) >= value)
            elif operator == "<":
                return df.filter(F.col(column) < value)
            elif operator == "<=":
                return df.filter(F.col(column) <= value)
            elif operator == "in":
                return df.filter(F.col(column).isin(value))
            elif operator == "not in":
                return df.filter(~F.col(column).isin(value))
            elif operator == "isNull":
                return df.filter(F.col(column).isNull())
            elif operator == "isNotNull":
                return df.filter(F.col(column).isNotNull())
            else:
                raise ValueError(f"Unsupported operator: {operator}")
        else:
            raise ValueError("Filter requires either 'condition' or 'column'/'value' parameters")
    
    def _transform_drop(self, df: DataFrame, params: Dict) -> DataFrame:
        """Drop specified columns."""
        cols = params.get("columns", [])
        return df.drop(*cols)
    
    def _transform_with_column(self, df: DataFrame, params: Dict) -> DataFrame:
        """Add a new column with an expression."""
        column_name = params.get("name")
        expr_type = params.get("type", "sql")
        expr = params.get("expr")
        
        if not column_name or not expr:
            raise ValueError("withColumn requires 'name' and 'expr' parameters")
        
        if expr_type == "sql":
            return df.withColumn(column_name, F.expr(expr))
        elif expr_type == "column":
            return df.withColumn(column_name, F.col(expr))
        elif expr_type == "literal":
            return df.withColumn(column_name, F.lit(expr))
        else:
            raise ValueError(f"Unsupported expression type: {expr_type}")
    
    def _transform_rename_column(self, df: DataFrame, params: Dict) -> DataFrame:
        """Rename a column."""
        existing = params.get("existing")
        new = params.get("new")
        
        if not existing or not new:
            raise ValueError("withColumnRenamed requires 'existing' and 'new' parameters")
        
        return df.withColumnRenamed(existing, new)
    
    def _transform_select_expr(self, df: DataFrame, params: Dict) -> DataFrame:
        """Select columns using SQL expressions."""
        exprs = params.get("expressions", [])
        return df.selectExpr(*exprs)
    
    def _transform_group_by(self, df: DataFrame, params: Dict) -> DataFrame:
        """Group by specified columns and apply aggregations."""
        columns = params.get("columns", [])
        aggs = params.get("agg", {})
        
        grouped = df.groupBy(*columns)
        
        # If aggregations are specified, apply them
        if aggs:
            agg_exprs = []
            for output_col, agg_config in aggs.items():
                agg_type = agg_config.get("type")
                input_col = agg_config.get("column")
                
                if agg_type == "count":
                    agg_exprs.append(F.count(input_col).alias(output_col))
                elif agg_type == "sum":
                    agg_exprs.append(F.sum(input_col).alias(output_col))
                elif agg_type == "avg":
                    agg_exprs.append(F.avg(input_col).alias(output_col))
                elif agg_type == "min":
                    agg_exprs.append(F.min(input_col).alias(output_col))
                elif agg_type == "max":
                    agg_exprs.append(F.max(input_col).alias(output_col))
                elif agg_type == "expr":
                    expr = agg_config.get("expr")
                    agg_exprs.append(F.expr(expr).alias(output_col))
                else:
                    raise ValueError(f"Unsupported aggregation type: {agg_type}")
            
            return grouped.agg(*agg_exprs)
        
        return grouped
    
    def _transform_aggregate(self, df: DataFrame, params: Dict) -> DataFrame:
        """Apply aggregations to a DataFrame."""
        aggs = params.get("aggregations", [])
        agg_exprs = []
        
        for agg in aggs:
            output_col = agg.get("as")
            agg_type = agg.get("type")
            input_col = agg.get("column")
            
            if agg_type == "count":
                agg_exprs.append(F.count(input_col).alias(output_col))
            elif agg_type == "sum":
                agg_exprs.append(F.sum(input_col).alias(output_col))
            elif agg_type == "avg":
                agg_exprs.append(F.avg(input_col).alias(output_col))
            elif agg_type == "min":
                agg_exprs.append(F.min(input_col).alias(output_col))
            elif agg_type == "max":
                agg_exprs.append(F.max(input_col).alias(output_col))
            elif agg_type == "expr":
                expr = agg.get("expr")
                agg_exprs.append(F.expr(expr).alias(output_col))
            else:
                raise ValueError(f"Unsupported aggregation type: {agg_type}")
        
        return df.agg(*agg_exprs)
    
    def _transform_join(self, df: DataFrame, params: Dict) -> DataFrame:
        """Join with another DataFrame."""
        other_df_alias = params.get("dataframe")
        join_type = params.get("type", "inner")
        join_cols = params.get("on", [])
        
        # We need a registry of dataframes to join from
        if not hasattr(self, "dataframe_registry"):
            raise ValueError("DataFrame registry not initialized. Cannot perform join.")
        
        if other_df_alias not in self.dataframe_registry:
            raise ValueError(f"DataFrame '{other_df_alias}' not found in registry")
        
        other_df = self.dataframe_registry[other_df_alias]
        
        if isinstance(join_cols, list):
            return df.join(other_df, on=join_cols, how=join_type)
        else:
            # Join expression as string
            return df.join(other_df, on=F.expr(join_cols), how=join_type)
    
    def _transform_union_by_name(self, df: DataFrame, params: Dict) -> DataFrame:
        """Union with another DataFrame by column names."""
        other_df_alias = params.get("dataframe")
        allow_missing_columns = params.get("allowMissingColumns", False)
        
        # We need a registry of dataframes for union
        if not hasattr(self, "dataframe_registry"):
            raise ValueError("DataFrame registry not initialized. Cannot perform union.")
        
        if other_df_alias not in self.dataframe_registry:
            raise ValueError(f"DataFrame '{other_df_alias}' not found in registry")
        
        other_df = self.dataframe_registry[other_df_alias]
        
        return df.unionByName(other_df, allowMissingColumns=allow_missing_columns)
    
    def _transform_with_window(self, df: DataFrame, params: Dict) -> DataFrame:
        """Apply window functions."""
        column_name = params.get("name")
        window_fn = params.get("function")
        partition_by = params.get("partitionBy", [])
        order_by = params.get("orderBy", [])
        
        window_spec = (
            F.window()
            .partitionBy(*partition_by)
            .orderBy(*[F.col(col) if isinstance(col, str) else col for col in order_by])
        )
        
        if window_fn == "row_number":
            return df.withColumn(column_name, F.row_number().over(window_spec))
        elif window_fn == "rank":
            return df.withColumn(column_name, F.rank().over(window_spec))
        elif window_fn == "dense_rank":
            return df.withColumn(column_name, F.dense_rank().over(window_spec))
        elif window_fn == "percent_rank":
            return df.withColumn(column_name, F.percent_rank().over(window_spec))
        elif window_fn == "lag":
            offset = params.get("offset", 1)
            default = params.get("default", None)
            col = params.get("column")
            return df.withColumn(column_name, F.lag(col, offset, default).over(window_spec))
        elif window_fn == "lead":
            offset = params.get("offset", 1)
            default = params.get("default", None)
            col = params.get("column")
            return df.withColumn(column_name, F.lead(col, offset, default).over(window_spec))
        else:
            raise ValueError(f"Unsupported window function: {window_fn}")
    
    def _transform_custom(self, df: DataFrame, params: Dict) -> DataFrame:
        """Apply a custom transformation using a function reference."""
        function_name = params.get("function")
        function_args = params.get("args", {})
        
        if not hasattr(self, "custom_functions"):
            raise ValueError("Custom functions registry not initialized")
        
        if function_name not in self.custom_functions:
            raise ValueError(f"Custom function '{function_name}' not found in registry")
        
        func = self.custom_functions[function_name]
        return func(df, **function_args)
    
    def _transform_repartition(self, df: DataFrame, params: Dict) -> DataFrame:
        """Repartition the DataFrame."""
        num_partitions = params.get("numPartitions")
        columns = params.get("columns", [])
        
        if columns:
            return df.repartition(num_partitions, *columns)
        else:
            return df.repartition(num_partitions)
    
    def _transform_coalesce(self, df: DataFrame, params: Dict) -> DataFrame:
        """Coalesce the DataFrame to fewer partitions."""
        num_partitions = params.get("numPartitions")
        return df.coalesce(num_partitions)
    
    def _transform_order_by(self, df: DataFrame, params: Dict) -> DataFrame:
        """Order the DataFrame by specified columns."""
        columns = params.get("columns", [])
        
        # Handle complex sorting with directions
        order_cols = []
        for col in columns:
            if isinstance(col, dict):
                col_name = col.get("column")
                direction = col.get("direction", "asc")
                
                if direction.lower() == "asc":
                    order_cols.append(F.col(col_name).asc())
                else:
                    order_cols.append(F.col(col_name).desc())
            else:
                order_cols.append(col)
        
        return df.orderBy(*order_cols)
    
    def _transform_limit(self, df: DataFrame, params: Dict) -> DataFrame:
        """Limit the number of rows in the DataFrame."""
        n = params.get("n")
        return df.limit(n)
    
    def _transform_sample(self, df: DataFrame, params: Dict) -> DataFrame:
        """Sample rows from the DataFrame."""
        fraction = params.get("fraction")
        seed = params.get("seed")
        withReplacement = params.get("withReplacement", False)
        
        if seed is not None:
            return df.sample(withReplacement=withReplacement, fraction=fraction, seed=seed)
        else:
            return df.sample(withReplacement=withReplacement, fraction=fraction)
    
    def _transform_distinct(self, df: DataFrame, params: Dict) -> DataFrame:
        """Get distinct rows."""
        return df.distinct()
    
    def _transform_cache(self, df: DataFrame, params: Dict) -> DataFrame:
        """Cache the DataFrame."""
        storage_level = params.get("storageLevel", "MEMORY_AND_DISK")
        
        # Map string to actual storage level
        import pyspark.storagelevel as SL
        storage_levels = {
            "MEMORY_ONLY": SL.StorageLevel.MEMORY_ONLY,
            "MEMORY_AND_DISK": SL.StorageLevel.MEMORY_AND_DISK,
            "MEMORY_ONLY_SER": SL.StorageLevel.MEMORY_ONLY_SER,
            "MEMORY_AND_DISK_SER": SL.StorageLevel.MEMORY_AND_DISK_SER,
            "DISK_ONLY": SL.StorageLevel.DISK_ONLY,
            "OFF_HEAP": SL.StorageLevel.OFF_HEAP,
        }
        
        level = storage_levels.get(storage_level, SL.StorageLevel.MEMORY_AND_DISK)
        return df.persist(level)
    
    def _transform_unpersist(self, df: DataFrame, params: Dict) -> DataFrame:
        """Unpersist the DataFrame."""
        blocking = params.get("blocking", False)
        df.unpersist(blocking)
        return df
    
    def _transform_checkpoint(self, df: DataFrame, params: Dict) -> DataFrame:
        """Checkpoint the DataFrame."""
        eager = params.get("eager", True)
        return df.checkpoint(eager)
    
    def _transform_drop_duplicates(self, df: DataFrame, params: Dict) -> DataFrame:
        """Drop duplicate rows."""
        subset = params.get("subset", None)
        
        if subset:
            return df.dropDuplicates(subset)
        else:
            return df.dropDuplicates()
    
    def _transform_fill_na(self, df: DataFrame, params: Dict) -> DataFrame:
        """Fill null values."""
        value = params.get("value")
        subset = params.get("subset", None)
        
        if subset:
            return df.fillna(value, subset)
        else:
            return df.fillna(value)
    
    def _transform_drop_na(self, df: DataFrame, params: Dict) -> DataFrame:
        """Drop rows with null values."""
        how = params.get("how", "any")
        thresh = params.get("thresh", None)
        subset = params.get("subset", None)
        
        return df.dropna(how=how, thresh=thresh, subset=subset)
    
    def _transform_explode(self, df: DataFrame, params: Dict) -> DataFrame:
        """Explode an array column to multiple rows."""
        column = params.get("column")
        return df.withColumn(column, F.explode(F.col(column)))
    
    def _transform_pivot(self, df: DataFrame, params: Dict) -> DataFrame:
        """Pivot the DataFrame."""
        pivot_col = params.get("column")
        values = params.get("values", None)
        
        pivot_df = df.groupBy().pivot(pivot_col, values)
        
        # Check if we need to apply aggregations
        aggs = params.get("agg", [])
        if aggs:
            agg_exprs = []
            for agg in aggs:
                agg_type = agg.get("type")
                col = agg.get("column")
                
                if agg_type == "sum":
                    agg_exprs.append(F.sum(col))
                elif agg_type == "avg":
                    agg_exprs.append(F.avg(col))
                elif agg_type == "min":
                    agg_exprs.append(F.min(col))
                elif agg_type == "max":
                    agg_exprs.append(F.max(col))
                elif agg_type == "count":
                    agg_exprs.append(F.count(col))
                else:
                    raise ValueError(f"Unsupported aggregation type: {agg_type}")
            
            return pivot_df.agg(*agg_exprs)
        
        return pivot_df
    
    # ---- Action Implementations ----
    
    def _action_collect(self, df: DataFrame, params: Dict) -> List[Dict]:
        """Collect DataFrame rows as a list of dictionaries."""
        return df.collect()
    
    def _action_count(self, df: DataFrame, params: Dict) -> int:
        """Count rows in the DataFrame."""
        return df.count()
    
    def _action_show(self, df: DataFrame, params: Dict) -> None:
        """Display DataFrame contents."""
        n = params.get("n", 20)
        truncate = params.get("truncate", True)
        vertical = params.get("vertical", False)
        
        df.show(n=n, truncate=truncate, vertical=vertical)
        return None
    
    def _action_to_pandas(self, df: DataFrame, params: Dict) -> Any:
        """Convert to Pandas DataFrame."""
        return df.toPandas()
    
    def _action_write(self, df: DataFrame, params: Dict) -> None:
        """Write DataFrame to storage."""
        format_type = params.get("format", "parquet")
        path = params.get("path")
        mode = params.get("mode", "overwrite")
        partition_by = params.get("partitionBy", None)
        
        writer = df.write.format(format_type).mode(mode)
        
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        
        # Handle options
        options = params.get("options", {})
        for key, value in options.items():
            writer = writer.option(key, value)
        
        writer.save(path)
        return None
    
    def _action_explain(self, df: DataFrame, params: Dict) -> str:
        """Explain the logical and physical plan."""
        extended = params.get("extended", False)
        mode = params.get("mode", "extended" if extended else "simple")
        
        return df.explain(mode)
    
    def _action_describe(self, df: DataFrame, params: Dict) -> DataFrame:
        """Compute summary statistics."""
        cols = params.get("cols", "*")
        return df.describe(cols)
    
    def _action_first(self, df: DataFrame, params: Dict) -> Any:
        """Get the first row."""
        return df.first()
    
    def _action_take(self, df: DataFrame, params: Dict) -> List:
        """Take n rows."""
        n = params.get("n", 1)
        return df.take(n)
    
    def _action_head(self, df: DataFrame, params: Dict) -> Any:
        """Get the first row(s)."""
        n = params.get("n", 1)
        return df.head(n)
    
    def _action_save_as_table(self, df: DataFrame, params: Dict) -> None:
        """Save as a table."""
        table_name = params.get("name")
        mode = params.get("mode", "overwrite")
        
        df.write.mode(mode).saveAsTable(table_name)
        return None
    
    # ---- DataFrame Registry Methods ----
    
    def init_dataframe_registry(self):
        """Initialize an empty DataFrame registry."""
        self.dataframe_registry = {}
        self.custom_functions = {}
    
    def register_dataframe(self, name: str, df: DataFrame):
        """Register a DataFrame in the registry."""
        if not hasattr(self, "dataframe_registry"):
            self.init_dataframe_registry()
        
        self.dataframe_registry[name] = df
        logger.info(f"Registered DataFrame: {name}")
    
    def register_custom_function(self, name: str, func: Callable):
        """Register a custom transformation function."""
        if not hasattr(self, "custom_functions"):
            self.init_dataframe_registry()
        
        self.custom_functions[name] = func
        logger.info(f"Registered custom function: {name}")
    
    def get_registered_dataframe(self, name: str) -> DataFrame:
        """Retrieve a registered DataFrame by name."""
        if not hasattr(self, "dataframe_registry"):
            raise ValueError("DataFrame registry not initialized")
        
        if name not in self.dataframe_registry:
            raise ValueError(f"DataFrame '{name}' not found in registry")
        
        return self.dataframe_registry[name]
    
    # ---- Utility Methods ----
    
    def resolve_column_references(self, df: DataFrame, column_refs: List) -> List:
        """
        Resolves column references to actual column objects.
        Handles both string column names and complex expressions.
        """
        resolved = []
        
        for ref in column_refs:
            if isinstance(ref, str):
                # Simple column reference
                resolved.append(F.col(ref))
            elif isinstance(ref, dict):
                # Expression
                expr_type = ref.get("type")
                
                if expr_type == "column":
                    resolved.append(F.col(ref.get("name")))
                elif expr_type == "literal":
                    resolved.append(F.lit(ref.get("value")))
                elif expr_type == "expr":
                    resolved.append(F.expr(ref.get("expr")))
                else:
                    raise ValueError(f"Unsupported column reference type: {expr_type}")
            else:
                # Assume it's already a Column
                resolved.append(ref)
        
        return resolved


# Helper functions for using the framework

def create_transformer(spark_session=None):
    """
    Create and initialize a DynamicTransformer instance.
    
    Args:
        spark_session: Active SparkSession
        
    Returns:
        Initialized DynamicTransformer
    """
    transformer = DynamicTransformer(spark_session)
    transformer.init_dataframe_registry()
    return transformer

def load_config_from_file(file_path):
    """
    Load transformation configuration from a JSON file.
    
    Args:
        file_path: Path to the JSON configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def apply_config(df, config, spark_session=None):
    """
    Apply a configuration to a DataFrame using the transformer.
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary or file path
        spark_session: Optional SparkSession
        
    Returns:
        Result of the transformation/action
    """
    transformer = create_transformer(spark_session)
    
    if isinstance(config, str):
        config = load_config_from_file(config)
    
    return transformer.transform_from_config(df, config)