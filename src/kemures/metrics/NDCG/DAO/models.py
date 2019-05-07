# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from apps.kemures.kernel.round.models import Round
from django.db import models


class NDCG(models.Model):
    round = models.ForeignKey(Round, unique=False, on_delete=models.CASCADE)
    value = models.FloatField()
    at = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
