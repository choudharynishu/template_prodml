from sqlalchemy import REAL, INTEGER, VARCHAR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from config.config import settings

class Base(DeclarativeBase):
    pass

class FMCG(Base):
    __tablename__ = settings.fmcg_table_name

    Date: Mapped[str] = mapped_column(VARCHAR(), primary_key=True)
    Product_Category: Mapped[str] = mapped_column(VARCHAR())
    Sales_Volume: Mapped[int] = mapped_column(INTEGER())
    Price: Mapped[float] = mapped_column(REAL())
    Promotion: Mapped[int] = mapped_column(INTEGER())
    Store_Location: Mapped[str] = mapped_column(VARCHAR())
    Weekday: Mapped[int] = mapped_column(INTEGER())
    Supplier_Cost: Mapped[float] = mapped_column(REAL())
    Replenishment_Lead_Time: Mapped[int] = mapped_column(INTEGER())
    Stock_level: Mapped[int] = mapped_column(INTEGER())
