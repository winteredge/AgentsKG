import asyncio
import os
import aiomysql
from dotenv import load_dotenv


load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

# --- 1. 配置你的数据库连接信息 ---
db_config = {
    "host": DB_HOST,
    "port": int(DB_PORT),
    "user": DB_USER,
    "password": DB_PASSWORD,
    "db": DB_NAME,
}

# 您的 CREATE TABLE 语句，从主脚本中复制而来
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS triples (
    id INT AUTO_INCREMENT PRIMARY KEY,
    subject VARCHAR(767) NOT NULL,
    predicate VARCHAR(767) NOT NULL,
    object TEXT NOT NULL,
    subject_id BIGINT NOT NULL,
    predicate_id BIGINT NOT NULL,
    object_id BIGINT NOT NULL,
    semantic_source VARCHAR(100) DEFAULT 'original',
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_triples_ids (subject_id, predicate_id, object_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


async def diagnose_lock_issue():
    """诊断数据库锁问题的专用脚本"""
    pool = None
    try:
        pool = await aiomysql.create_pool(minsize=1, maxsize=1, **db_config)
        print("--- 1. 成功连接到数据库 ---")

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:

                # 设置当前会话的锁等待超时为 5 秒，以便快速失败
                print("\n--- 2. 设置当前会话的锁等待超时为 5 秒 ---")
                await cursor.execute("SET INNODB_LOCK_WAIT_TIMEOUT = 5;")
                print("设置成功。")

                print("\n--- 3. 尝试执行 CREATE TABLE IF NOT EXISTS... ---")
                try:
                    await cursor.execute(CREATE_TABLE_SQL)
                    print("--- 🥳 CREATE TABLE 命令成功执行！---")
                    print("这表明锁问题已经消失了。您可以尝试运行主脚本了。")
                    return  # 成功了就直接结束

                except aiomysql.MySQLError as e:
                    if e.args[0] == 1205:  # 确认是锁等待超时错误
                        print("\n--- ❌ 错误复现：Lock wait timeout exceeded! ---")
                        print("错误信息:", e)
                        print("\n--- 4. 尝试查询 InnoDB 锁信息 ---")

                        # 查询正在等待的锁
                        print(
                            "\n----- A. 查询当前正在等待的锁 (information_schema.INNODB_LOCK_WAITS) -----"
                        )
                        try:
                            await cursor.execute(
                                """
                                SELECT
                                    r.trx_id AS waiting_trx_id,
                                    r.trx_mysql_thread_id AS waiting_thread,
                                    r.trx_query AS waiting_query,
                                    b.trx_id AS blocking_trx_id,
                                    b.trx_mysql_thread_id AS blocking_thread,
                                    (SELECT query FROM information_schema.processlist WHERE id = b.trx_mysql_thread_id) AS blocking_query
                                FROM
                                    information_schema.INNODB_LOCK_WAITS w
                                JOIN
                                    information_schema.INNODB_TRX b ON b.trx_id = w.blocking_trx_id
                                JOIN
                                    information_schema.INNODB_TRX r ON r.trx_id = w.requesting_trx_id;
                            """
                            )
                            lock_waits = await cursor.fetchall()
                            if lock_waits:
                                print("找到了锁等待信息：")
                                for row in lock_waits:
                                    print(row)
                            else:
                                print("在 INNODB_LOCK_WAITS 中未找到明确的等待关系。")
                        except aiomysql.MySQLError as e_lock:
                            print(f"查询 INNODB_LOCK_WAITS 失败: {e_lock}")

                        # 查询所有活动的事务
                        print(
                            "\n----- B. 查询所有活动的事务 (information_schema.INNODB_TRX) -----"
                        )
                        try:
                            await cursor.execute(
                                "SELECT * FROM information_schema.INNODB_TRX ORDER BY trx_started;"
                            )
                            transactions = await cursor.fetchall()
                            if transactions:
                                print("找到了以下活动事务：")
                                for trx in transactions:
                                    print(
                                        f"  - Trx ID: {trx['trx_id']}, State: {trx['trx_state']}, Started: {trx['trx_started']}, Thread ID: {trx['trx_mysql_thread_id']}"
                                    )
                                print(
                                    "\n请重点关注那些 'trx_state' 为 RUNNING 且已启动（Started）很久的事务。"
                                )
                                print(
                                    "找到它的 'trx_mysql_thread_id'，这就是需要被 KILL 的进程 ID。"
                                )
                            else:
                                print("未找到任何活动的 InnoDB 事务。")
                        except aiomysql.MySQLError as e_trx:
                            print(f"查询 INNODB_TRX 失败: {e_trx}")

                    else:
                        print(f"\n--- ❌ 发生了其他数据库错误 ---")
                        print("错误信息:", e)

    except Exception as ex:
        print(f"\n--- 发生未知错误 ---")
        print(f"错误信息: {ex}")
    finally:
        if pool:
            pool.close()
            await pool.wait_closed()
            print("\n--- 诊断脚本执行完毕，连接已关闭 ---")


if __name__ == "__main__":
    asyncio.run(diagnose_lock_issue())
