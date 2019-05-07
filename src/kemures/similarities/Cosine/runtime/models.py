# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from apps.kemures.kernel.round.models import Round
from django.db import models


class CosineSimilarityRunTime(models.Model):
    round = models.OneToOneField(Round, unique=True, on_delete=models.CASCADE)
    started_at = models.DateTimeField()
    finished_at = models.DateTimeField()
