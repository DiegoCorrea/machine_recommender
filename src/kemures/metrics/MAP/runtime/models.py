# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from apps.kemures.metrics.MAP.DAO.models import MAP
from django.db import models


class MAPRunTime(models.Model):
    id = models.OneToOneField(MAP, primary_key=True, on_delete=models.CASCADE)
    started_at = models.DateTimeField()
    finished_at = models.DateTimeField()
