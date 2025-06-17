import psycopg2
from config import load_config
from colum_config import colum_config

def create_tables():
    """Create tables in the PostgreSQL database from column_config"""
    commands = []
    table_commands = {}

    for section in colum_config.values():
        for column, properties in section.items():
            table_name = properties['table_name']
            if table_name not in table_commands:
                table_commands[table_name] = []

            data_type = properties['db_data_type'].upper()
            col_definition = f"{column} {data_type}"

            # Add length to VARCHAR if not already present
            if data_type == 'VARCHAR' and '(' not in col_definition:
                col_definition += "(255)"

            if not properties['db_is_null']:
                col_definition += " NOT NULL"

            table_commands[table_name].append(col_definition)

    for table_name, columns in table_commands.items():
        create_command = f"CREATE TABLE IF NOT EXISTS {table_name} (\n  " + ",\n  ".join(columns) + "\n);"
        commands.append(create_command)

    try:
        config = load_config()
        with psycopg2.connect(**config) as conn:
            with conn.cursor() as cur:
                for command in commands:
                    print(f"Executing SQL:\n{command}\n")  # Debug print
                    cur.execute(command)
    except (psycopg2.DatabaseError, Exception) as error:
        print(f"Database error: {error}")

if __name__ == '__main__':
    create_tables()
