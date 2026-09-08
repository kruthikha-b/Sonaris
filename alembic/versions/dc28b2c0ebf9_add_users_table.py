"""add users table

Revision ID: dc28b2c0ebf9
Revises: 2eb2f6cdbee8
Create Date: 2026-09-08 15:58:01.077547

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "dc28b2c0ebf9"
down_revision: Union[str, Sequence[str], None] = "2eb2f6cdbee8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass