import pandas as pd
import sqlglot
import sqlglot.expressions as exp
from typing import Dict, Set, List, Any, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("sqlglot").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class SQLMetadataExtractor:
    def __init__(self, dialect: Optional[str] = None):
        """
        Initializes the extractor.

        Args:
            dialect (Optional[str]): The SQL dialect to use for parsing (e.g., 'snowflake', 'bigquery', 'postgres').
                                     If None, sqlglot will try to guess.
        """
        self.source_code_columns: Set[str] = {'AR_SRCE_CDE', 'DATA_SRCE_CDE'}
        self.dialect = dialect

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
            try:
                parsed_expressions = sqlglot.parse(sql_str, read=self.dialect)
                if not parsed_expressions:
                    logger.warning("SQL string parsed into an empty expression list.")
                    return {}
                parsed = parsed_expressions[0]
            except Exception as parse_error:
                logger.error(f"sqlglot failed to parse SQL: {parse_error}\nQuery snippet: {sql_str[:500]}...")
                return {}

            # Data Structures
            alias_to_table_map: Dict[str, str] = {}
            table_details: Dict[str, Dict[str, Set[str]]] = {}
            cte_names: Set[str] = set()

            # --- 1. Find CTEs ---
            if isinstance(parsed, exp.Select) and parsed.ctes:
                for cte in parsed.ctes:
                    if isinstance(cte, exp.CTE) and cte.alias:
                        cte_names.add(cte.alias.lower())
                        # Process CTE recursively
                        cte_metadata = self._extract_sql_metadata(cte.this.sql())
                        table_details[cte.alias.lower()] = cte_metadata.get(cte.alias.lower(), {"columns": set(), "source_codes": set()})

            logging.debug(f"Found CTE names: {cte_names}")

            # --- 2. Find Tables and Aliases (excluding CTEs) ---
            for table_expr in parsed.find_all(exp.Table):
                table_name_parts = []
                if table_expr.db:
                    table_name_parts.append(table_expr.db.lower())
                table_name_parts.append(table_expr.name.lower())
                full_table_name = ".".join(table_name_parts)

                if full_table_name in cte_names:
                    if table_expr.alias:
                        alias_to_table_map[table_expr.alias.lower()] = full_table_name
                    continue

                alias = table_expr.alias.lower() if table_expr.alias else full_table_name
                if full_table_name not in table_details:
                    table_details[full_table_name] = {'columns': set(), 'source_codes': set()}
                alias_to_table_map[alias] = full_table_name
                if alias != full_table_name:
                    alias_to_table_map[full_table_name] = full_table_name

            logging.debug(f"Found non-CTE tables: {list(table_details.keys())}")
            logging.debug(f"Alias map: {alias_to_table_map}")

            # --- 3. Find Columns ---
            processed_col_keys: Set[Tuple[str, str]] = set()
            for col_expr in parsed.find_all(exp.Column):
                table_ref = col_expr.table.lower() if col_expr.table else None
                column_name_upper = col_expr.name.upper()

                if not table_ref:
                    if len(alias_to_table_map) == 1:
                        table_ref = list(alias_to_table_map.keys())[0]
                    else:
                        continue

                actual_table = alias_to_table_map.get(table_ref)
                if actual_table and actual_table in table_details:
                    col_key = (actual_table, column_name_upper)
                    if col_key not in processed_col_keys:
                        table_details[actual_table]['columns'].add(column_name_upper)
                        processed_col_keys.add(col_key)
                        logging.debug(f"Found column: {actual_table}.{column_name_upper}")

            # --- 4. Find Source Codes in WHERE Clauses ---
            for where_clause in parsed.find_all(exp.Where):
                for condition in where_clause.find_all((exp.EQ, exp.In)):
                    column_ref_node = None
                    value_nodes = []

                    if isinstance(condition.left, exp.Column) and condition.left.table:
                        column_ref_node = condition.left
                        if isinstance(condition, exp.EQ):
                            if isinstance(condition.right, exp.Literal):
                                value_nodes.append(condition.right)
                        elif isinstance(condition, exp.In) and isinstance(condition.right, (exp.Tuple, exp.List)):
                            value_nodes.extend(condition.right.expressions)

                    elif isinstance(condition.right, exp.Column) and condition.right.table:
                        if isinstance(condition, exp.EQ) and isinstance(condition.left, exp.Literal):
                            column_ref_node = condition.right
                            value_nodes.append(condition.left)

                    if not column_ref_node:
                        continue

                    col_name_upper = column_ref_node.name.upper()
                    if col_name_upper in self.source_code_columns:
                        table_ref = column_ref_node.table.lower()
                        actual_table = alias_to_table_map.get(table_ref)

                        if actual_table and actual_table in table_details:
                            for val_node in value_nodes:
                                if isinstance(val_node, exp.Literal):
                                    extracted_value = str(val_node.literal_value)
                                    table_details[actual_table]['source_codes'].add(extracted_value)
                                    logging.debug(f"Found source code: {actual_table}.{col_name_upper} -> {extracted_value}")

            # --- 5. Propagate CTE Metadata ---
            for cte_name in cte_names:
                if cte_name in alias_to_table_map:
                    actual_table = alias_to_table_map[cte_name]
                    if actual_table in table_details:
                        table_details[actual_table]['columns'].update(table_details[cte_name]['columns'])
                        table_details[actual_table]['source_codes'].update(table_details[cte_name]['source_codes'])

            # --- 6. Final Formatting ---
            final_result = {
                table: data for table, data in table_details.items()
                if data['columns'] or data['source_codes']
            }
            for table_data in final_result.values():
                table_data['columns'] = sorted(list(table_data['columns']))
                table_data['source_codes'] = sorted(list(table_data['source_codes']))

            logging.debug(f"Final extracted metadata: {final_result}")
            return final_result

        except Exception as e:
            logger.error(f"Unexpected error processing SQL query: {e}\nQuery: {sql_str[:500]}...", exc_info=True)
            return {}

    def _update_master_dict(self, master_dict: Dict[str, Dict[str, Set[str]]], current_metadata: Dict[str, Dict[str, List[str]]]):
        """Update master dictionary with new metadata. Expects lists from extraction, converts to sets."""
        for table, metadata in current_metadata.items():
            if table not in master_dict:
                master_dict[table] = {'columns': set(), 'source_codes': set()}
            master_dict[table]['columns'].update(metadata.get('columns', []))
            master_dict[table]['source_codes'].update(metadata.get('source_codes', []))

    def process_dataframe(self, df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
        """Process all queries from a DataFrame."""
        if 'query_text' not in df.columns:
            raise ValueError("DataFrame must contain 'query_text' column")

        master_dict_sets: Dict[str, Dict[str, Set[str]]] = {}

        total_rows = len(df)
        for index, row in df.iterrows():
            query = row['query_text']
            if pd.isna(query) or not isinstance(query, str) or not query.strip():
                logger.warning(f"Skipping empty or invalid query at index {index} (Row {index + 1}/{total_rows})")
                continue

            logger.info(f"Processing query at index {index} (Row {index + 1}/{total_rows})...")
            current_metadata = self._extract_sql_metadata(query)
            if current_metadata:
                self._update_master_dict(master_dict_sets, current_metadata)
                logger.info(f"Successfully processed query at index {index} (Row {index + 1}/{total_rows})")
            else:
                logger.warning(f"No metadata extracted for query at index {index} (Row {index + 1}/{total_rows})")

        final_master_dict: Dict[str, Dict[str, List[str]]] = {}
        for table, data in master_dict_sets.items():
            final_master_dict[table] = {
                'columns': sorted(list(data['columns'])),
                'source_codes': sorted(list(data['source_codes']))
            }
        logger.info(f"Finished processing DataFrame. Found metadata for {len(final_master_dict)} tables.")
        return final_master_dict

def test_queries():
    """Test the SQL metadata extraction with various cases."""
    extractor = SQLMetadataExtractor()

    test_cases = [
        {
            "name": "Simple Select",
            "sql": "SELECT t1.colA, t1.DATA_SRCE_CDE FROM db.schema.table1 t1 WHERE t1.DATA_SRCE_CDE = 'ABC'"
        },
        {
            "name": "Simple Join",
            "sql": """
                SELECT t1.colA, t2.colB, t1.DATA_SRCE_CDE, t2.AR_SRCE_CDE
                FROM table1 t1
                JOIN table2 t2 ON t1.id = t2.fk_id
                WHERE t1.DATA_SRCE_CDE IN ('A1', 'B2') AND t2.AR_SRCE_CDE = 'X1'
            """
        }
    ]

    master_results_sets: Dict[str, Dict[str, Set[str]]] = {}

    for i, test in enumerate(test_cases):
        print(f"\n=== Running Test {i+1}: {test['name']} ===")
        try:
            result = extractor._extract_sql_metadata(test['sql'])
            print("\nExtracted Metadata (Isolated):")
            if not result:
                print("No metadata extracted.")
            printable_result = {}
            for table, metadata in result.items():
                printable_result[table] = {
                    'columns': sorted(list(metadata.get('columns', set()))),
                    'source_codes': sorted(list(metadata.get('source_codes', set())))
                }

            for table, metadata in printable_result.items():
                print(f"\nTable: {table}")
                print(f"  Columns: {metadata['columns']}")
                print(f"  Source Codes: {metadata['source_codes']}")

            extractor._update_master_dict(master_results_sets, result)

        except Exception as e:
            print(f"\nError processing test case '{test['name']}': {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print("\n\n=== Final Accumulated Results ===")
    final_master_dict: Dict[str, Dict[str, List[str]]] = {}
    for table, data in master_results_sets.items():
        final_master_dict[table] = {
            'columns': sorted(list(data['columns'])),
            'source_codes': sorted(list(data['source_codes']))
        }

    for table, metadata in final_master_dict.items():
        print(f"\nTable: {table}")
        print(f"  Columns: {metadata['columns']}")
        print(f"  Source Codes: {metadata['source_codes']}")

if __name__ == "__main__":
    test_queries()