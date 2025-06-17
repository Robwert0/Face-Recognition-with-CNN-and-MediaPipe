colum_config = {
    "faculty": {
        "faculty_id": {
            "db_data_type": "SERIAL",
            "db_is_null": False,
            "table_name": "faculty"
        },
        "faculty_name": {
            "db_data_type": "VARCHAR",
            "db_is_null": False,
            "table_name": "faculty"
        }
    },
    "profile": {
        "profile_id": {
            "db_data_type": "SERIAL",
            "db_is_null": False,
            "table_name": "profile"
        },
        "profile_name": {
            "db_data_type": "VARCHAR",
            "db_is_null": False,
            "table_name": "profile"
        },
        "faculty_id": {
            "db_data_type": "INTEGER",
            "db_is_null": False,
            "table_name": "profile"
        }
    },
    "course": {
        "course_id": {
            "db_data_type": "SERIAL",
            "db_is_null": False,
            "table_name": "course"
        },
        "course_name": {
            "db_data_type": "VARCHAR",
            "db_is_null": False,
            "table_name": "course"
        }
    },
    "profile_course": {
        "profile_id": {
            "db_data_type": "INTEGER",
            "db_is_null": False,
            "table_name": "profile_course"
        },
        "course_id": {
            "db_data_type": "INTEGER",
            "db_is_null": False,
            "table_name": "profile_course"
        }
    },
    "person": {
        "person_id": {
            "db_data_type": "SERIAL",
            "db_is_null": False,
            "table_name": "person"
        },
        "full_name": {
            "db_data_type": "VARCHAR",
            "db_is_null": False,
            "table_name": "person"
        },
        "course_id": {
            "db_data_type": "INTEGER",
            "db_is_null": False,
            "table_name": "person"
        },
        "profile_id": {
            "db_data_type": "INTEGER",
            "db_is_null": True,
            "table_name": "person"
        }
    },
    "attendance_log": {
        "log_id": {
            "db_data_type": "SERIAL",
            "db_is_null": False,
            "table_name": "attendance_log"
        },
        "person_id": {
            "db_data_type": "INTEGER",
            "db_is_null": False,
            "table_name": "attendance_log"
        },
        "date": {
            "db_data_type": "DATE",
            "db_is_null": False,
            "table_name": "attendance_log"
        },
        "time": {
            "db_data_type": "TIME",
            "db_is_null": False,
            "table_name": "attendance_log"
        },
        "accuracy": {
            "db_data_type": "NUMERIC(5,2)",
            "db_is_null": True,
            "table_name": "attendance_log"
        }
    }
}
