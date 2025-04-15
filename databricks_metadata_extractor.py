import pandas as pd
import sqlglot
import sqlglot.expressions as exp
from typing import Dict, Set, List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SQLMetadataExtractor:
    def __init__(self):
        """
        Initializes the extractor with Databricks dialect.
        """
        self.source_code_columns: Set[str] = {'AR_SRCE_CDE', 'DATA_SRCE_CDE'}
        self.dialect = 'databricks'

    def _extract_sql_metadata(self, sql_str: str) -> Dict[str, Dict[str, Set[str]]]:
        """
        Extract metadata from SQL query using sqlglot AST traversal.

        Returns a dictionary mapping actual table names (lowercase, potentially schema-qualified
        if schema is used consistently) to their extracted columns and source codes.
        """
        if not sql_str or not isinstance(sql_str, str):
            logger.warning("Received empty or non-string SQL input.")
            return {}

        try:
            # Parse the SQL into an AST
            parsed = sqlglot.parse_one(sql_str, read=self.dialect)
            if not parsed:
                logger.warning("SQL string parsed into an empty expression.")
                return {}

            # Data Structures
            alias_to_table_map: Dict[str, str] = {}
            table_details: Dict[str, Dict[str, Set[str]]] = {}
            cte_names: Set[str] = set()

            # --- 1. Find CTEs ---
            for cte in parsed.find_all(exp.CTE):
                if cte.alias:
                    cte_name = cte.alias.lower()
                    cte_names.add(cte_name)
                    # Process CTE recursively
                    cte_metadata = self._extract_sql_metadata(cte.this.sql())
                    table_details[cte_name] = cte_metadata.get(cte_name, {"columns": set(), "source_codes": set()})

            logger.debug(f"Found CTE names: {cte_names}")

            # --- 2. Find Tables and Aliases ---
            for table_expr in parsed.find_all(exp.Table):
                table_name = table_expr.name.lower()
                if table_expr.db:
                    table_name = f"{table_expr.db.lower()}.{table_name}"

                if table_name in cte_names:
                    continue

                alias = table_expr.alias.lower() if table_expr.alias else table_name
                alias_to_table_map[alias] = table_name
                if table_name not in table_details:
                    table_details[table_name] = {"columns": set(), "source_codes": set()}

            logger.debug(f"Alias map: {alias_to_table_map}")

            # --- 3. Find Columns ---
            for col_expr in parsed.find_all(exp.Column):
                table_ref = col_expr.table.lower() if col_expr.table else None
                column_name = col_expr.name.lower()

                if not table_ref:
                    if len(alias_to_table_map) == 1:
                        table_ref = list(alias_to_table_map.keys())[0]
                    else:
                        continue

                actual_table = alias_to_table_map.get(table_ref)
                if actual_table and actual_table in table_details:
                    table_details[actual_table]["columns"].add(column_name)

            # --- 4. Find Source Codes in WHERE Clauses ---
            for where_clause in parsed.find_all(exp.Where):
                for condition in where_clause.find_all((exp.EQ, exp.In)):
                    column_ref = None
                    values = []

                    if isinstance(condition.left, exp.Column):
                        column_ref = condition.left
                        if isinstance(condition.right, exp.Literal):
                            values.append(condition.right)
                        elif isinstance(condition.right, exp.Tuple):
                            values.extend(condition.right.expressions)

                    elif isinstance(condition.right, exp.Column):
                        column_ref = condition.right
                        if isinstance(condition.left, exp.Literal):
                            values.append(condition.left)

                    if column_ref and column_ref.name.upper() in self.source_code_columns:
                        table_ref = column_ref.table.lower() if column_ref.table else None
                        actual_table = alias_to_table_map.get(table_ref)
                        if actual_table and actual_table in table_details:
                            for value in values:
                                if isinstance(value, exp.Literal):
                                    table_details[actual_table]["source_codes"].add(value.this)

            # --- 5. Propagate CTE Metadata ---
            for cte_name in cte_names:
                if cte_name in alias_to_table_map:
                    actual_table = alias_to_table_map[cte_name]
                    if actual_table in table_details:
                        table_details[actual_table]["columns"].update(table_details[cte_name]["columns"])
                        table_details[actual_table]["source_codes"].update(table_details[cte_name]["source_codes"])

            # --- 6. Final Formatting ---
            final_result = {
                table: {
                    "columns": sorted(list(data["columns"])),
                    "source_codes": sorted(list(data["source_codes"])),
                }
                for table, data in table_details.items()
                if data["columns"] or data["source_codes"]
            }

            logger.debug(f"Final extracted metadata: {final_result}")
            return final_result

        except Exception as e:
            logger.error(f"Unexpected error processing SQL query: {e}\nQuery: {sql_str[:500]}...", exc_info=True)
            return {}

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process all queries from a DataFrame and return a DataFrame with the extracted metadata.
        
        Returns:
            pd.DataFrame: A DataFrame with columns:
                - table_name: The name of the table
                - columns: All columns found for the table
                - source_code_values: All source code values found for the table
        """
        if "query_text" not in df.columns:
            raise ValueError("DataFrame must contain 'query_text' column")

        master_dict_sets: Dict[str, Dict[str, Set[str]]] = {}

        total_rows = len(df)
        for index, row in df.iterrows():
            query = row["query_text"]
            if pd.isna(query) or not isinstance(query, str) or not query.strip():
                logger.warning(f"Skipping empty or invalid query at index {index} (Row {index + 1}/{total_rows})")
                continue

            logger.info(f"Processing query at index {index} (Row {index + 1}/{total_rows})...")
            current_metadata = self._extract_sql_metadata(query)
            if current_metadata:
                for table, metadata in current_metadata.items():
                    if table not in master_dict_sets:
                        master_dict_sets[table] = {"columns": set(), "source_codes": set()}
                    master_dict_sets[table]["columns"].update(metadata.get("columns", []))
                    master_dict_sets[table]["source_codes"].update(metadata.get("source_codes", []))
                logger.info(f"Successfully processed query at index {index} (Row {index + 1}/{total_rows})")
            else:
                logger.warning(f"No metadata extracted for query at index {index} (Row {index + 1}/{total_rows})")

        # Convert to DataFrame
        result_data = []
        for table, data in master_dict_sets.items():
            result_data.append({
                "table_name": table,
                "columns": sorted(list(data["columns"])),
                "source_code_values": sorted(list(data["source_codes"]))
            })

        result_df = pd.DataFrame(result_data)
        logger.info(f"Finished processing DataFrame. Found metadata for {len(result_df)} tables.")
        return result_df 