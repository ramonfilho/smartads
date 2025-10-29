"""
Configuração do banco de dados PostgreSQL
Gerencia conexão com Cloud SQL e operações CRUD
"""

import os
from sqlalchemy import create_engine, Column, Integer, String, Text, TIMESTAMP, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

# Base para modelos SQLAlchemy
Base = declarative_base()

# =============================================================================
# MODELO: Lead CAPI
# =============================================================================

class LeadCAPI(Base):
    """Modelo para leads capturados com dados CAPI"""
    __tablename__ = 'leads_capi'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Identificação
    email = Column(String(255), nullable=False, index=True)
    name = Column(String(255))
    phone = Column(String(50))

    # Dados CAPI
    fbp = Column(String(255))
    fbc = Column(String(255))
    event_id = Column(String(255), unique=True, index=True)

    # Tracking
    user_agent = Column(Text)
    client_ip = Column(String(50))
    event_source_url = Column(Text)

    # UTMs
    utm_source = Column(String(255))
    utm_medium = Column(String(255))
    utm_campaign = Column(String(255))
    utm_term = Column(String(255))
    utm_content = Column(String(255))

    # Outros
    tem_comp = Column(String(50))

    # Timestamps
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    def to_dict(self) -> Dict:
        """Converte para dict"""
        return {
            'id': self.id,
            'email': self.email,
            'name': self.name,
            'phone': self.phone,
            'fbp': self.fbp,
            'fbc': self.fbc,
            'event_id': self.event_id,
            'user_agent': self.user_agent,
            'client_ip': self.client_ip,
            'event_source_url': self.event_source_url,
            'utm_source': self.utm_source,
            'utm_medium': self.utm_medium,
            'utm_campaign': self.utm_campaign,
            'utm_term': self.utm_term,
            'utm_content': self.utm_content,
            'tem_comp': self.tem_comp,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

# =============================================================================
# CONFIGURAÇÃO DA ENGINE
# =============================================================================

def get_database_url() -> str:
    """
    Retorna URL de conexão com o banco

    Ordem de prioridade:
    1. DATABASE_URL (env var completa)
    2. Componentes individuais (DB_HOST, DB_NAME, etc)
    3. Fallback para SQLite local (desenvolvimento)
    """
    # Opção 1: URL completa
    if os.getenv('DATABASE_URL'):
        return os.getenv('DATABASE_URL')

    # Opção 2: Componentes individuais (Cloud SQL)
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'smart_ads')
    db_user = os.getenv('DB_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD')

    if db_host and db_password:
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    # Opção 3: Fallback SQLite (desenvolvimento/testes)
    logger.warning("Usando SQLite (desenvolvimento) - Configure PostgreSQL para produção")
    return "sqlite:////tmp/smart_ads_dev.db"

def get_engine():
    """Cria engine SQLAlchemy"""
    database_url = get_database_url()

    # SQLite precisa de configuração especial
    if database_url.startswith('sqlite'):
        return create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            echo=False
        )

    # PostgreSQL
    return create_engine(
        database_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        echo=False
    )

# Engine global
engine = get_engine()

# SessionLocal para dependency injection
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_database():
    """Inicializa database (cria tabelas se não existirem)"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database inicializado com sucesso")
        return True
    except Exception as e:
        logger.error(f"❌ Erro ao inicializar database: {e}")
        return False

def get_db() -> Session:
    """
    Dependency para FastAPI
    Uso: db: Session = Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =============================================================================
# OPERAÇÕES CRUD
# =============================================================================

def create_lead_capi(db: Session, lead_data: Dict) -> LeadCAPI:
    """Cria novo lead no banco"""
    lead = LeadCAPI(**lead_data)
    db.add(lead)
    db.commit()
    db.refresh(lead)
    return lead

def get_lead_by_email(db: Session, email: str) -> Optional[LeadCAPI]:
    """Busca lead por email (mais recente)"""
    return db.query(LeadCAPI).filter(LeadCAPI.email == email).order_by(LeadCAPI.created_at.desc()).first()

def get_lead_by_event_id(db: Session, event_id: str) -> Optional[LeadCAPI]:
    """Busca lead por event_id"""
    return db.query(LeadCAPI).filter(LeadCAPI.event_id == event_id).first()

def get_leads_by_emails(db: Session, emails: List[str]) -> List[LeadCAPI]:
    """Busca múltiplos leads por email (batch)"""
    return db.query(LeadCAPI).filter(LeadCAPI.email.in_(emails)).all()

def get_recent_leads(db: Session, limit: int = 100) -> List[LeadCAPI]:
    """Retorna leads mais recentes"""
    return db.query(LeadCAPI).order_by(LeadCAPI.created_at.desc()).limit(limit).all()

def count_leads(db: Session) -> int:
    """Conta total de leads"""
    return db.query(LeadCAPI).count()

def count_leads_with_fbp(db: Session) -> int:
    """Conta leads com FBP preenchido"""
    return db.query(LeadCAPI).filter(LeadCAPI.fbp.isnot(None)).count()

def count_leads_with_fbc(db: Session) -> int:
    """Conta leads com FBC preenchido"""
    return db.query(LeadCAPI).filter(LeadCAPI.fbc.isnot(None)).count()
