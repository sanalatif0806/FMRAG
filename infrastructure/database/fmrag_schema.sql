-- FMRAG unified database schema
-- Merges Client_ver8 (em_client_test.sql) + Broker_ver8 (em_broker_test.sql)
-- Adds FL round tracking tables for FMRAG
-- MySQL 8.0+ compatible (InnoDB throughout, utf8mb4)
-- Run on every node: mysql -u root -p < fmrag_schema.sql

CREATE DATABASE IF NOT EXISTS fmrag
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE fmrag;

-- ── Cloudlet resource registry ───────────────────────────────────────────────
-- One row per cloudlet (broker and clients both write here)
CREATE TABLE IF NOT EXISTS resource (
  ip_address      VARCHAR(45)   NOT NULL,
  cloudlet_name   VARCHAR(100)  NOT NULL,
  status          VARCHAR(45)   DEFAULT 'online',
  last_updated    DATETIME      DEFAULT CURRENT_TIMESTAMP
                                ON UPDATE CURRENT_TIMESTAMP,
  cpu_cores       INT           DEFAULT 0,
  cpu_avload      FLOAT         DEFAULT 0,
  memory_total    FLOAT         DEFAULT 0,
  memory_free     FLOAT         DEFAULT 0,
  memory_used     FLOAT         DEFAULT 0,
  disk_total      FLOAT         DEFAULT 0,
  disk_free       FLOAT         DEFAULT 0,
  bytes_sent      BIGINT        DEFAULT 0,
  bytes_recv      BIGINT        DEFAULT 0,
  PRIMARY KEY (ip_address)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── Broker resource scoring ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS decision_parameters (
  cloudlet_ip     VARCHAR(45)     NOT NULL,
  resource_index  DECIMAL(5,4)    DEFAULT 0,
  resource_level  VARCHAR(45)     DEFAULT 'normal',
  PRIMARY KEY (cloudlet_ip),
  FOREIGN KEY (cloudlet_ip) REFERENCES resource(ip_address)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── Users ──────────────────────────────────────────────────────────────────--
CREATE TABLE IF NOT EXISTS ituser (
  id          INT           NOT NULL AUTO_INCREMENT,
  user        VARCHAR(100)  NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,   -- bcrypt hash (NOT plaintext)
  created_at  DATETIME      DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── VM / workload requests ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS user_request (
  request_number  INT           NOT NULL AUTO_INCREMENT,
  request_dt      DATETIME      DEFAULT CURRENT_TIMESTAMP,
  ip_address      VARCHAR(45)   NOT NULL,   -- source cloudlet IP
  cloudlet_name   VARCHAR(100),
  user            VARCHAR(100)  NOT NULL,
  vm_name         VARCHAR(200),
  vm_ip           VARCHAR(225),
  vm_cpu          INT           DEFAULT 1,
  vm_storage      FLOAT         DEFAULT 20.0,
  vm_memory       FLOAT         DEFAULT 4.0,
  vm_user         VARCHAR(100),
  vm_file_name    VARCHAR(200),
  PRIMARY KEY (request_number),
  FOREIGN KEY (ip_address) REFERENCES resource(ip_address)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── Request lifecycle status ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS status (
  id               INT         NOT NULL AUTO_INCREMENT,
  request_number   INT         NOT NULL,
  ip_address       VARCHAR(45) DEFAULT NULL,  -- source cloudlet IP (matches original schema)
  request_status   VARCHAR(45) DEFAULT 'incomplete',
  decision_status  VARCHAR(45) DEFAULT 'pending',
  offload_status   VARCHAR(45) DEFAULT 'pending',
  migration_status VARCHAR(45) DEFAULT 'pending',
  error_status     VARCHAR(10) DEFAULT NULL,
  mes_to_user      VARCHAR(200) DEFAULT NULL,
  updated_at       DATETIME    DEFAULT CURRENT_TIMESTAMP
                               ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  FOREIGN KEY (request_number) REFERENCES user_request(request_number)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── Broker offload decisions ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS decision (
  request_number  INT         NOT NULL,
  src_cl_ip       VARCHAR(45),
  dst_cl_ip       VARCHAR(45),
  decision_sent   VARCHAR(45) DEFAULT 'no',
  decided_at      DATETIME    DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (request_number),
  FOREIGN KEY (request_number) REFERENCES user_request(request_number)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── Request timing log ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS times (
  id                       INT         NOT NULL AUTO_INCREMENT,
  request_number           INT         NOT NULL,
  vm_actual_file_upload_time VARCHAR(45),
  vm_file_size             VARCHAR(45),
  PRIMARY KEY (id),
  FOREIGN KEY (request_number) REFERENCES user_request(request_number)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── CIPYZ: eligible cloudlet staging table ────────────────────────────────────
-- Temporary staging used by broker each poll cycle.
-- Populated by process_pending_requests(), cleared at start of each cycle.
-- Mirrors original CIPYZ el_cloudlet table exactly.
CREATE TABLE IF NOT EXISTS el_cloudlet (
  id              INT         NOT NULL AUTO_INCREMENT,
  cloudlet_ip     VARCHAR(45) NOT NULL,   -- source cloudlet that made the request
  request_number  INT         NOT NULL,
  el_cloudlet_ip  VARCHAR(45) NOT NULL,   -- eligible destination IP (or 'NULL')
  PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── FMRAG: FL round tracking ──────────────────────────────────────────────────
-- Records each FL round: which clients participated, LoRA delta size,
-- aggregated AUROC after round. Used by evaluation/run_evaluation.py.
CREATE TABLE IF NOT EXISTS fl_round (
  round_id        INT           NOT NULL AUTO_INCREMENT,
  round_number    INT           NOT NULL,
  started_at      DATETIME      DEFAULT CURRENT_TIMESTAMP,
  completed_at    DATETIME      DEFAULT NULL,
  n_clients       INT           DEFAULT 0,
  global_auroc    FLOAT         DEFAULT NULL,
  comm_cost_mb    FLOAT         DEFAULT NULL,
  PRIMARY KEY (round_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── FMRAG: per-client FL participation ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS fl_client_round (
  id              INT           NOT NULL AUTO_INCREMENT,
  round_id        INT           NOT NULL,
  client_ip       VARCHAR(45)   NOT NULL,
  n_samples       INT           DEFAULT 0,
  local_auroc     FLOAT         DEFAULT NULL,
  delta_norm      FLOAT         DEFAULT NULL,   -- L2 norm of LoRA delta
  training_sec    FLOAT         DEFAULT NULL,
  PRIMARY KEY (id),
  FOREIGN KEY (round_id) REFERENCES fl_round(round_id),
  FOREIGN KEY (client_ip) REFERENCES resource(ip_address)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── FMRAG: drug hypothesis log ────────────────────────────────────────────────
-- Stores generated hypotheses (no patient identifiers)
CREATE TABLE IF NOT EXISTS drug_hypothesis (
  id              INT           NOT NULL AUTO_INCREMENT,
  generated_at    DATETIME      DEFAULT CURRENT_TIMESTAMP,
  client_ip       VARCHAR(45),
  drug_cui        VARCHAR(20),
  drug_name       VARCHAR(200),
  drug_class      VARCHAR(100),
  score           FLOAT,
  contraindicated TINYINT(1)    DEFAULT 0,
  narrative       TEXT,
  PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── FMRAG: CAG cache statistics ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cag_stats (
  id              INT           NOT NULL AUTO_INCREMENT,
  recorded_at     DATETIME      DEFAULT CURRENT_TIMESTAMP,
  client_ip       VARCHAR(45),
  cached_patients INT           DEFAULT 0,
  total_hits      INT           DEFAULT 0,
  retrieval_latency_ms FLOAT    DEFAULT NULL,
  PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
