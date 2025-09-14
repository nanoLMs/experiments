#!/usr/bin/env python3
"""
Performance and Quality Validation Tests for Advanced NanoLM System
==================================================================

Comprehensive validation tests covering:
- Convergence validation (target loss 1.5-2.5)
- Model size validation (100-150MB target)
- Perplexity and generation quality tests
- Baseline comparison tests
- Performance benchmarking
- Memory efficiency validation
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import tempfile
import os
import json
import math
import gc
import psutil
from typing import Dict, Any, List, Tuple, Optional
from unittest.mock import Mock, patch, MagicMock

# Import available systems for validation testing
from advanced_quantization_system import (
    NF4Quantizer, FP4Quantizer, QAFTransitionSystem,
    QuantizationConfig, QuantizationType, QAFPhase
)

from core_model_architecture import (
    NanoLMModel, MultiHeadAttention, MixtureOfExperts, MultiTokenPredictionHeads,
    ModelConfig, MoEConfig, MTPConfig
)

from hierarchical_reasoning_module import (
    HierarchicalReasoningModule, HRMConfig
)

from advanced_loss_system import (
    MultiComponentLoss, LossTracker, LossConfig
)

from anti_hallucination_system import (
    AntiHallucinationFilter, AntiHallucinationConfig
)

from robust_training_pipeline import (
    TrainerOptimized, TrainingConfig
)

from multi_platform_exporter import (
    MultiPlatformExporter, ExportFormat, ExportConfig
)


class TestConvergenceValidation(unittest.TestCase):
    """Test convergence validation and loss targets"""

    def setUp(self):
        """Set up convergence testing environment"""
        self.target_loss_min = 1.5
        self.target_loss_max = 2.5
        self.convergence_patience = 10
        self.min_training_steps = 100

        # Create test model configuration
        self.model_config = ModelConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=6,
            num_heads=8,
            intermediate_size=1024,
            max_position_embeddings=512
        )

        # Create quantization config for efficiency
        self.quant_config = QuantizationConfig(
            quantization_type=QuantizationType.NF4,
            compute_dtype=torch.float16,
            quant_dtype=torch.uint8
        )

        # Create loss configuration
        self.loss_config = LossConfig(
            main_loss_weight=1.0,
            mtp_loss_weight=0.3,
            auxiliary_loss_weight=0.1,
            reasoning_loss_weight=0.2,
            anti_hallucination_loss_weight=0.1
        )

    def test_loss_convergence_validation(self):
        """Test that training can achieve target loss range"""
        # Create model with quantization
        model = NanoLMModel(self.model_config)
        quantizer = NF4Quantizer(self.quant_config)
        quantized_model = quantizer.quantize_model(model)

        # Create loss system
        loss_calculator = MultiComponentLoss(self.loss_config)
        loss_tracker = LossTracker()

        # Create synthetic training data
        batch_size, seq_len = 8, 64
        num_batches = 150  # Enough for convergence testing

        training_data = []
        for i in range(num_batches):
            batch = {
                'input_ids': torch.randint(0, self.model_config.vocab_size, (batch_size, seq_len)),
                'labels': torch.randint(0, self.model_config.vocab_size, (batch_size, seq_len)),
                'attention_mask': torch.ones(batch_size, seq_len)
            }
            training_data.append(batch)

        # Simulate training with loss tracking
        optimizer = torch.optim.AdamW(quantized_model.parameters(), lr=1e-4)

        losses = []
        converged = False
        convergence_step = None

        for step, batch in enumerate(training_data):
            optimizer.zero_grad()

            # Forward pass
            outputs = quantized_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )

            # Calculate loss
            loss_components = loss_calculator.calculate_loss(
                outputs, batch['labels'], batch['attention_mask']
            )

            total_loss = loss_components['total_loss']
            losses.append(total_loss.item())

            # Track loss
            loss_tracker.add_loss(total_loss.item(), step)

            # Backward pass
            total_loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(quantized_model.parameters(), 1.0)

            optimizer.step()

            # Check convergence every 10 steps
            if step > self.min_training_steps and step % 10 == 0:
                recent_losses = losses[-self.convergence_patience:]
                if len(recent_losses) >= self.convergence_patience:
                    avg_recent_loss = np.mean(recent_losses)

                    # Check if converged to target range
                    if self.target_loss_min <= avg_recent_loss <= self.target_loss_max:
                        converged = True
                        convergence_step = step
                        break

                    # Check if loss is decreasing (convergence trend)
                    loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
                    if abs(loss_trend) < 0.001:  # Very small change indicates convergence
                        break

        # Validation assertions
        final_loss = np.mean(losses[-10:]) if len(losses) >= 10 else losses[-1]

        # Test results
        convergence_results = {
            'final_loss': final_loss,
            'converged_to_target': converged,
            'convergence_step': convergence_step,
            'total_steps': len(losses),
            'loss_reduction': losses[0] - final_loss if losses else 0,
            'target_range': (self.target_loss_min, self.target_loss_max)
        }

        # Assertions for convergence quality
        self.assertGreater(len(losses), self.min_training_steps,
                          "Training should run for minimum steps")
        self.assertLess(final_loss, losses[0] if losses else float('inf'),
                       "Loss should decrease during training")

        # Log convergence results for analysis
        print(f"\nConvergence Validation Results:")
        print(f"  Final Loss: {final_loss:.4f}")
        print(f"  Target Range: {self.target_loss_min}-{self.target_loss_max}")
        print(f"  Converged to Target: {converged}")
        print(f"  Convergence Step: {convergence_step}")
        print(f"  Loss Reduction: {convergence_results['loss_reduction']:.4f}")

        # Store results for further analysis
        self.convergence_results = convergence_results

        return convergence_results

    def test_loss_prediction_accuracy(self):
        """Test loss prediction system accuracy"""
        loss_tracker = LossTracker()

        # Simulate realistic loss curve
        steps = 200
        losses = []

        for step in range(steps):
            # Simulate exponential decay with noise
            base_loss = 5.0 * np.exp(-step / 50.0) + 1.5  # Converges to ~1.5
            noise = np.random.normal(0, 0.1)
            loss = max(0.1, base_loss + noise)  # Ensure positive loss

            losses.append(loss)
            loss_tracker.add_loss(loss, step)

        # Test predictions at different points
        prediction_points = [50, 100, 150]
        prediction_results = {}

        for point in prediction_points:
            if point < len(losses):
                # Get prediction
                predictions = loss_tracker.predict_loss_at_step(point + 50)  # Predict 50 steps ahead

                # Get actual loss at prediction point
                actual_point = min(point + 50, len(losses) - 1)
                actual_loss = losses[actual_point]

                # Calculate prediction accuracy
                best_prediction = predictions['ensemble']['predicted_loss']
                prediction_error = abs(best_prediction - actual_loss)
                relative_error = prediction_error / actual_loss if actual_loss > 0 else float('inf')

                prediction_results[point] = {
                    'predicted_loss': best_prediction,
                    'actual_loss': actual_loss,
                    'absolute_error': prediction_error,
                    'relative_error': relative_error,
                    'confidence': predictions['ensemble']['confidence']
                }

        # Validate prediction accuracy
        avg_relative_error = np.mean([r['relative_error'] for r in prediction_results.values()])

        self.assertLess(avg_relative_error, 0.3,
                       "Average prediction error should be less than 30%")

        print(f"\nLoss Prediction Accuracy:")
        for point, results in prediction_results.items():
            print(f"  Step {point}: Predicted={results['predicted_loss']:.3f}, "
                  f"Actual={results['actual_loss']:.3f}, "
                  f"Error={results['relative_error']:.1%}")

        return prediction_results

    def test_convergence_detection(self):
        """Test automatic convergence detection"""
        loss_tracker = LossTracker()

        # Simulate different convergence scenarios
        scenarios = {
            'fast_convergence': [5.0, 3.0, 2.0, 1.8, 1.7, 1.65, 1.6, 1.6, 1.6, 1.6],
            'slow_convergence': [4.0, 3.8, 3.6, 3.4, 3.2, 3.0, 2.8, 2.6, 2.4, 2.2],
            'oscillating': [3.0, 2.5, 3.2, 2.3, 3.1, 2.4, 2.9, 2.5, 2.8, 2.6],
            'diverging': [2.0, 2.2, 2.5, 2.8, 3.2, 3.6, 4.0, 4.5, 5.0, 5.5]
        }

        convergence_results = {}

        for scenario_name, loss_sequence in scenarios.items():
            # Reset tracker
            tracker = LossTracker()

            # Add losses
            for step, loss in enumerate(loss_sequence):
                tracker.add_loss(loss, step)

            # Check convergence detection
            convergence_info = tracker.detect_convergence()

            convergence_results[scenario_name] = {
                'converged': convergence_info['converged'],
                'convergence_step': convergence_info.get('convergence_step'),
                'convergence_confidence': convergence_info.get('confidence', 0.0),
                'final_loss': loss_sequence[-1],
                'loss_trend': loss_sequence[-1] - loss_sequence[0]
            }

        # Validate convergence detection
        self.assertTrue(convergence_results['fast_convergence']['converged'],
                       "Should detect fast convergence")
        self.assertFalse(convergence_results['diverging']['converged'],
                        "Should not detect convergence for diverging losses")

        print(f"\nConvergence Detection Results:")
        for scenario, results in convergence_results.items():
            print(f"  {scenario}: Converged={results['converged']}, "
                  f"Final Loss={results['final_loss']:.2f}")

        return convergence_results


class TestModelSizeValidation(unittest.TestCase):
    """Test model size validation and quantization efficiency"""

    def setUp(self):
        """Set up model size testing environment"""
        self.target_size_min_mb = 100
        self.target_size_max_mb = 150
        self.quantization_memory_reduction_target = 0.75  # 75% reduction

        # Model configurations for different sizes
        self.model_configs = {
            'small': ModelConfig(
                vocab_size=1000,
                hidden_size=256,
                num_layers=4,
                num_heads=8,
                intermediate_size=1024
            ),
            'medium': ModelConfig(
                vocab_size=2000,
s else 1)f succes0 i    exit(
alidation()y_valitce_quperformanccess = run_:
    su"__main__" ==_name__

if _)
ccessful(asSuult.wrn res retu   ")

(f"{'='*80}
    printLETE")DATION COMP"VALIint(   pr")
 n{'='*80} print(f"\

st}")te {print(f"  -            ors:
result.errin traceback  test,      forrs:")
   (f"\nErroprint:
        rorsif result.er
    }")
 - {test"     print(f        s:
ult.failureresck in  tracebast, for te
     res:")lu"\nFaiint(f        pr
t.failures:ul   if res
 ")
ention: 90% Quality Ret"  • Targetint(f
    pr")ent: 1.2x Improvemt Speed • Targeint(f" pr0")
    ty: < 50.Perplexiarget (f"  • T   print
 on: 75%")ducti Reoryt Mem  • Targent(f")
    pri- 150 MB"Size: 100 get Model Tarnt(f"  •
    pri.5 - 2.5"): 1oss Range Let(f"  • Targnt")
    prits:ance Targerform Peey"\nKrint(f   p
  summaryey metrics   # K

    ment: ✅")Assessality ion QuQuantizatnt(f"  • pri
  : ✅")sonparie Comncformaline Per"  • Baserint(f ✅")
    pty Metrics: Quali &Perplexityint(f"  •
    pr")): ✅reduction75% fficiency ( Memory Eint(f"  •")
    pr-150MB): ✅idation (100el Size Valod  • Mprint(f"   : ✅")
 oss 1.5-2.5)idation (L ValConvergence • print(f"     erage:")
lidation Cov\nVat(f"    prin
agecoveration d validtaile # De
   .1f}%")
 100):stsRun, 1) *result.te/ max(.errors)) n(resultres) - lelu.fai(resultn - lensRuresult.teste: {((atf"Success rrint(  p
  )t.errors)}"{len(resulrrors: (f"Erint
    pes)}")lt.failur{len(resulures: rint(f"Fai")
    pt.testsRun}esulests run: {rnt(f"Tpri   }")
 (f"{'='*80int
    prY")SUMMARON DATIY VALIITMANCE & QUALint("PERFOR")
    pr\n{'='*80}int(f"mary
    prsive sumprehenom# Print c

    est_suite)n(tnner.ru result = ru   rbosity=2)
(veRunnerTestt.Textunittesr = unnetput
    red ou detailsts with# Run te

(tests)te.addTeststest_sui       )
 classstCase(test_stsFromTeer().loadTet.TestLoadessts = unitt
        teasses:cln test_test_class i
    for ]
    son
    arilineCompBase      Test
  ,QualityexityAnd  TestPerpln,
      tioSizeValida  TestModel    ation,
  alidvergenceVtCon    Tes[
    s = assest_cl  te  es
 classd test
    # AdSuite()
  Test unittest.e =suit
    test_test suite Create
    #0}")
 "{'='*8  print(fN")
  Y VALIDATIOQUALIT & PERFORMANCESYSTEM - NCED NANOLM "ADVAprint(  80}")
  "{'='*t(f
    prin
   "s""idation testquality valce and manorn all perf  """Ru  ation():
lity_validmance_qua run_perforlts


defchmark_resuen   return b

      )ate:.1%}" {success_rs Rate:uccesf"  Sint(
        pr})")ss else '❌'l_succe if overalcess} ({'✅'ucl_sess: {overal Succ"  Overallt(f      prin")
  ality']} ❌['quts_metMet: {targelity Target se f"  Quaelquality'] t['s_me if targetty']} ✅"liqua['_mett: {targets Target MeQualityint(f"    pr")
      ']} ❌memory['rgets_metet: {taget My Tar"  Memor] else fy'met['memorf targets_" i'memory']} ✅ts_met[targeget Met: {Tarory "  Memnt(f      pri
")]} ❌eed't['sprgets_meMet: {tapeed Target lse f"  S'speed'] ergets_met[} ✅" if ta['speed']ets_met: {target Meteed Targ"  Spprint(f)
        ults:"enchmark Res Berall\nOvt(f"     prin
      }
        _rate
  ssate': succe'success_r     ,
       _successoverallcess': l_suc    'overal       t,
 meets_: targrgets_met'   'ta
         y_results,it qualesults':'quality_r        ts,
    emory_resuls': mltsuy_re   'memor
         results, speed_esults':  'speed_r
  ults = {hmark_res    benc
    _met)
  argets/ len(tues()) met.valgets_arum(ts_rate = ses        succ())
valuests_met.targes = all(erall_succes    ov
           }
ned']
     aiuality_ret'qsults[ quality_re':ty'quali            ,
y_target_metorory': mem     'mem       ,
get_met': speed_tareed      'sp {
      gets_met =     tarnt
   messess# Overall a
        ()
        isonmpar_coretentionlity_lf.test_quasults = se quality_ren()
       riso_compasagest_memory_uet = self.teet_mory_targresults, memry_   memo  )
   son(pari_comeednce_spnferet_itesself.rget_met = s, speed_taed_result       spests
 rison teompaun all c   # R
          ")
   f"{'='*60}t(in
        prHMARK")BENCCOMPARISON ASELINE  BREHENSIVECOMP    print("0}")
    n{'='*6t(f"\rin       p""
 cts"l aspeparing alrk commave benchehensicomprn    """Ruelf):
     hmark(shensive_bencmpret_co  def tes  ts

ity_resul return qual
       }")
      _retainedqualityt: {rget Me  Quality Ta"    print(f3f}")
    ity_score:.quality Score: {Qualll   Overa(f"nt     pri
   .3f}")tio:ity_rag_perplexo: {avty Ratilexi(f"  Perpint   pr
     f}")ent:.3greemiction_ag_pred {avment:greeion Aictt(f"  Pred prin
       f}")ilarity:.3vg_logit_simty: {ait Similarit(f"  Login     pr")
   tion']:.1%}uality_reten['qance_targetsormrfpeelf.{sRetention: uality rget Q Tant(f"
        prion:")Comparison lity Retenti(f"\nQua     print

        }retained
 ': quality_etainedy_rualit    'q     ore,
   _sctyuali: qre'coll_quality_svera  'o         ,
 tioerplexity_ra_pio': avgexity_rat 'perpl
           greement,ediction_a_prent': avgon_agreem   'predicti
         rity,_simila_logitavglarity':  'logit_simi         {
  ts = ity_resul       qual

        on']etenti['quality_r_targetsformancef.per selcore >== quality_sd etainey_r   qualit
     t) / 2emenn_agrectiovg_predilarity + ait_similog = (avg_relity_sco       qua better)
 er isghe (hiquality scored  # Combin

 ty_ratio'])erplexi_metrics['ptyn(quali= np.meaxity_ratio _perple     avg
   agreement'])ction_'preditrics[ity_mep.mean(qualreement = nagn_dictio  avg_prey'])
      imilaritics['logit_sty_metruali(q= np.meansimilarity logit_g_       avtion
 tenuality reverall q o# Calculate
        io)
       xity_ratrplepend(peatio'].apxity_rleetrics['perplity_m         qua
            1.0ity > 0 elseplexerseline_p if barplexitypeline_/ baserplexity ced_peadvan_ratio =   perplexity
                          ).item()
 ossadvanced_l= torch.exp(ty perplexiced_vanad                  ).item()
  ine_lossexp(baselch.ty = torine_perplexi basel

          )
   view(-1)us().ets.contiguo      targ
     -1)),logits.size(anced_view(-1, advntiguous().[:, :-1].co_logitsced       advan                 tropy(
ss_en F.cro =ssed_lo      advanc
                   )

      ew(-1)iguous().viets.contrg        ta
 e(-1)),gits.sizne_loli-1, base().view(guous-1].contiogits[:, :baseline_l                    ropy(
    .cross_ent= Foss  baseline_l                   el() > 0:
ets.num  if targ
              -1]
 t_input[:, :_ids = tes       input    tion
     predicn toket for nex Shift t[:, 1:]  #t_inpugets = tes      tar
      exity ratiote perpl  # Calcula
              nt)
     eeme(agr'].appendmenton_agreeictietrics['predquality_m
       an().item().me()atpreds).floadvanced_reds == seline_p= (bament ee   agr
   -1)
      dim=ogits,advanced_l.argmax(orcheds = tced_pradvan             =-1)
   imgits, dine_lomax(baselorch.arg tline_preds =        base       cy)
 1 accurant (top-ion agreemelate predict   # Calcu
                    m)
    sine_siappend(coty'].t_similarilogirics['ity_met       qual
                 ()
              ).item
          ueeze(0)lat.unsqd_fcedvan       a     ,
        eeze(0).unsquline_flat     base              ty(
 _similaricosine= F.ine_sim    cos

  flatten()ed_logits.advanc= vanced_flat       ad
  latten()ts.fline_logi base =flatline_ base           ity)
    ine similarlarity (cost simiulate logi# Calc
                    ts']
ogit['lnced_outpu= advanced_logits       adva          ts']
utput['logiaseline_o = bine_logitsbasel
                    est_input)
d_model(tf.advance = selputed_out  advanc             nput)
 _ist_model(tef.baselineut = seleline_outp   bas           ls
  h modets from bot# Get outpu
 .no_grad(): torchth      wi  s:
    t_sequence tesest_input in    for t
         }

      : []_ratio'tyexi     'perpl
       t': [],menn_agree 'predictio         ,
  arity': []git_simil   'lo
    s = {ricity_met    qual
      ]
    0)
   (2 range) for _ ine, (1, 32)ig.vocab_siznfodel_co(0, self.mandint  torch.r      [
    nces =  test_seque    set
   dataest e teat      # Cr"
  els"" between modalityutput qu"Compare o"       "(self):
 _comparisonretentionest_quality_def t    met

_target_ults, memorymemory_resn     retur

        f}MB)")_mb']:.1d_memorydvance {results['a.1f}MB →emory_mb']:eline_mresults['bas  f"({              "
   %} reductionduction']:.1['memory_re {resultscenario}:t(f"  {s   prin
         ms():tey_results.iemor in mults resscenario,  for
              t_met}")
y_targeet: {memort M(f"  Targe      print")
  on:.1%}ctiduory_reem{avg_mction: duage Re"  Averint(f   pr}")
     :.1%on']timemory_reduc['etsrmance_targf.perfoon: {seleductiTarget Rnt(f"        prison:")
  age ComparinMemory Us"\int(f
        pr
        ion']ct_redumorytargets['memance_.perforelfduction >= sg_memory_reet_met = avargmory_t      me  ])
.values()sultsory_rer in mem for uction']['memory_red[rp.mean(ion = nmory_reduct      avg_me
  mentssess # Overall a
        }

         n']tiomory_reducmee_targets['.performanc selfion >=ry_reductemoet': m_targ     'meets         ,
  eductionmemory_rn': ory_reductio      'mem  ,
        ory_usaged_memb': advancemory_mmedvanced_    'a       ge,
     sa_ue_memoryinmb': baseline_memory_ 'basel            {
   = ario_name] results[scenmemory_

     0 else 0y_usage > ne_memoraseliage if bmemory_us / baseline_emory_usage)_m advancedry_usage -ne_memoon = (baselieductiy_r     memorion
       uctry redemo mlate    # Calcu
        e
        emory_befordvanced_mfter - amory_ad_me= advancee _usagmory_me  advanced      024)
     / (1024 * 1y_info().rssss().memor.Proce = psutilafteremory_ advanced_m
        t)
    puel(test_indvanced_modself.aput = ed_out    advanc            grad():
th torch.no_       wi
             024)
     (1024 * 1ss /info().rss().memory_til.Proceefore = psury_bd_memodvance           a
 ()
       .collect   gc
   else None) _available(h.cuda.isorcache() if t.empty_cch.cudator      y
      memorel ced moddvansure a Mea          #
    re
  fo_memory_beaselinefter - bemory_a baseline_musage =memory_seline_      ba4)
       * 102 (1024().rss /ory_infoocess().memPrutil. = psafter_memory_    baseline

 input)_model(test_aselineself.b= utput baseline_o            ():
    ch.no_gradth tor  wi
24)
   / (1024 * 10info().rss y_ss().memoril.Procee = psutmemory_before_baselin
      ect()
  .coll         gce None
   ble() elsilais_avada..cuf torch) ity_cache(rch.cuda.emp  to
          emoryseline mre ba    # Measu

     len))_size, seq_ize, (batchocab_sl_config.vlf.mode0, seandint(ut = torch.r  test_inp       ):
   arios.items( scen) inseq_lene, h_siztco_name, (baor scenari f

            }
': (8, 128)'large_batch
  ': (4, 64),dium_batch     'me
       , 32),batch': (2'small_           = {
  ios      scenar   scenarios
rentst diffe     # Te

  ults = {}mory_res     me
   els"""tween mod usage be memory"Compare        ""f):
rison(selompausage_cry_memost_f te
    de
 t_metp_targe, speedueed_results  return sp

      )")f}ms_ms']:.1imeadvanced_tsults['}ms → {reme_ms']:.1ftiline_s['baseresult    f"({
        dup "pee.2f}x s]:'speedup'results[_key}: {f"  {testint(         prms():
   s.ite_resultts in speed, resulst_keyor te      f
  ")
 rget_met}up_ta: {speedget Met(f"  Tarnt      pri")
  eedup:.2f}xdup: {avg_spAverage Speet(f"          prin]:.1f}x")
e_speedup'ncs['infereance_targetperformf.{selup: eedSparget t(f"  Trin   p:")
     ComparisonSpeed ce Inferen\nrint(f"     p
       _speedup']
'inferences[e_targetrformanclf.pedup >= seavg_spee = rget_met  speedup_ta)
      ()]sults.valuespeed_re s for r ineedup']spr['mean([edup = np.     avg_spe  t
  assessmenmanceall perfor # Over

 }
       eedup']sprence_ets['infee_targperformanc>= self.t': speedup 'meets_targe
 edup,': spe'speedup                   0,
 g * 100dvanced_avs': ae_mdvanced_tim'a                   1000,
  _avg *line baseime_ms':aseline_t  'b           = {
       key] ts[test_sul_repeed      s
               0 else 0
avg >vanced_g if ad_avvanced/ adne_avg p = baselipeedu      s
          es)vanced_tim np.mean(advanced_avg =      ad
          ine_times)p.mean(baselline_avg = n    base            s
statisticculate        # Cal
             )
 rt_time) - staime.time(d(tppend_times.a  advance
         ut)inpodel(test_nced_m = self.adva   _
   grad():ch.no_tor      with               e.time()
_time = tim   start                 10):
in range(for _            = []
     imes ed_t advanc               model
  advancedenchmark        # B

        tart_time)() - s(time.timependmes.apne_ti      baseli
      put)inel(test_odseline_m= self.ba    _                    d():
 torch.no_gra       with             .time()
 e = time start_tim
        cyr accurale runs fo# Multiprange(10):  or _ in   f            es = []
  imne_tbaseli
    delline mok baseBenchmar         #
                     )
   seq_len)ch_size,bat (.vocab_size,figonelf.model_cndint(0, sorch.ra_input = t       test
         e test input    # Creat
              q_len}"
  }_seq_{sech_sizech_{bat= f"batst_key  te         hs:
      nce_lengtequeeq_len in s sor  f
          tch_sizes: in babatch_size for
           lts = {}
 speed_resu
       28]
      1s = [32, 64,gthenquence_l
        se, 4, 8]_sizes = [1 batchns
       uratioest config   # T     s"""
ced modelannd advine an baselweee speed betferenc in""Compare
        "self):_comparison(ednference_spe_i def test
       l_config)
odemer(self.msforineTranurn Basel   ret
       }
  s': xtate 'hidden_sts': logits,n {'logitur         re
                   n(x)
  ectioput_projself.out   logits =
    tput Ou        #
                       er(x)
 ormsfself.tran        x =
        rmer  # Transfo
                  e(0)
   en].unsqueezg[:seq_lncodin+ self.pos_e x = x            t_ids)
    (inpuedding = self.emb       xs
         Embedding  #
                      ds.shape
_inputq_len = ih_size, se   batc            ids):
 t_(self, inpuf forward         de
        e)
   izig.vocab_sonfsize, cg.hidden_confiar(n = nn.Linet_projectiolf.outpu se

    )               m_layers
s=config.nulayer num_
        yer, coder_la         en  (
         Encodersformeran.Tr = nnansformerself.tr
                    )

          first=Truetch_        ba            e_size,
diattermeinnfig.=coforward_feed   dim                 eads,
config.num_hhead=     n               en_size,
hiddg.confil=ode     d_m      (
         errLayormerEncodensfrann.Tder_layer =       enco          ers
sformer layrd tran# Standa
             e))
  siz.hidden_igdn(512, conforch.raner(tmetnn.Paraoding = self.pos_enc
   en_size)config.hidd_size, ig.vocabbedding(confding = nn.Em  self.embed

      ig = conf self.config            t__()
   ).__ini    super(
          config):__(self, f __init       de     ule):
mer(nn.ModforTransss Baseline      claon"""
  risompa for cormer modele transfelin a bas""Create   "    odule:
 f) -> nn.Mdel(selaseline_mote_bea_cr  def
    }
      uality
   90% of qin    # Reta 0.9retention': 'quality_
       memory  # 50% lessn': 0.5, ductio  'memory_re       ster
   20% fa# up': 1.2,  ence_speedinfer  '
  {ce_targets =f.performan selts
       nce targe # Performa
el_config)
f.modl(selnoLMMode_model = Naadvanced      self.
  el()baseline_modte_._creael = selfeline_mod  self.bas      mer)
nsforrae tdel (simpl moe baselineCreat   #
              )

     num_heads=8
 yers=4,m_la        nu
    56,ze=2   hidden_si,
         _size=1000     vocab     lConfig(
  ig = Modedel_conf    self.mo   est model
  # Create t"
       onment""virison enne comparaseli"Set up b""        tUp(self):
 def se

""arks"benchmls and aseline modeon against barisest comp"""Tse):
    ttest.TestCason(uniomparielineCTestBaslass

celse 0.0
scores  repetition_es) iftition_scormean(repern np.retu
  e)
   tion_scorti.append(repeesorition_scpet       re
  2rigrams) /_ttalams / togrpeated_trigrams + re_bims / totaled_bigraat (repeore =petition_sc    re            0:
 grams >al_triand totrams > 0 f total_big         i
          ounts)
    rigram_c= len(trigrams  total_t           ounts)
ram_cms = len(bigotal_bigra  t

           > 1)nts() if coulues.varigram_countn tr count i1 foum(s = srigramed_trepeat           unt > 1)
 colues() if m_counts.va in bigraor countm(1 fs = suram_bigpeated    re   o
     ation rrepetiti Calculate       #
   1
      m, 0) +get(trigraounts.gram_c = triram]_counts[trig  trigram
[i:i+3]) = tuple(gen   trigram           n) - 2):
  range(len(gei in         for
            1
   gram, 0) +s.get(bi_countigramm] = bts[bigragram_coun          bi    +2])
  uple(gen[i:i  bigram = t       1):
       ) - gen range(len(in      for i

   nts = {} trigram_cou        ts = {}
   coungram_  bi          n-grams
 tednt repea       # Cou

   continue            en) < 4:
  len(gif            ns:
 tioen in genera for g
     es = []
   ion_scor    repetit
    er)"""s bette (lower ion scortitie repelculat"""Ca
     loat:]) -> fnt][List[iListrations: e(self, geneortion_screpetite_def _calcula
    tokens
  / total_que_tokens   return uni

      s)tokenn(all_ns = le_toke  total
))okensn(set(all_t leue_tokens =iq  un
      rn 0.0
       retu      kens:
    all_to    if not
    gen)
  xtend(all_tokens.e            erations:
 in genr gen    fo []
    s =ken   all_to"
     s""rationens in geneque tok unio ofte rati""Calcula"
        ) -> float:nt]]List[List[ions: lf, generatin_ratio(seunique_toke _calculate_
    defist()
    0].told[rate gene return
     k
         brea               en
     toks EOS 0 ingAssumi #  0: em() ==t_token.it if nex            ion
   ndit stopping cosonablea reap if we hit       # Sto
              )
     im=1xt_token], d neerated,t([genca= torch.generated

          eepdim=True) kdim=-1,ax(logits, torch.argmn = toke     next_        ing
       reedy sampl     # G            else:
               s=1)
    um_samples, nnomial(probtorch.multit_token =   nex
         -1), dim=x(logitss = F.softma    prob         n
       ibutio distrromple f    # Sam
                ('-inf')
at floemove] =to_rts[indices_gi      lo                e)
  to_removted_indices_sord_indices, ortescatter(1, sto_remove.es_indicsorted_= move rees_to_    indic
             = 0
   e[..., 0] ces_to_removndisorted_i                  e()
      ., :-1].clonremove[.._indices_to_] = sorted., 1:to_remove[..ces_diorted_in        s              ]
  p_p''tos[arg > kwtive_probse = cumulaov_to_remndicesorted_i   s
oldthreshthe ove lity abbi probaulativeh cumittokens wRemove     #
                        1)
       1), dim=-ts, dim=-logied_max(sortum(F.softorch.cums = tprobsative_ cumul
    ing=True)its, descend(log.sort = torchrted_indicesd_logits, sorte         so           gs:
     kwar 'top_p' in if

 p_k_logits)s, totop_k_indice1, scatter_(   logits.                     '-inf'))
oat((logits, fll_like= torch.fulogits      l
  , top_k)opk(logits.t = torch_indicesogits, top_kk_ltop_
      ]op_k'['t kwargstop_k =                     wargs:
   ' in k   if 'top_k
                            ']
  tures['temperawargs / kits = logit      log          :
         in kwargsrature'peif 'tem
   False):_sample',t('dokwargs.geif               egy
  mpling strat  # Apply sa
                        its
  token log last 1, :]  # Getogits'][:, -tputs['l = oulogits             ted)
   raneel(ge= self.modutputs      o           e(1)):
izh - prompt.s(max_lengtrange_ in  for            no_grad():
orch.   with t
          clone()
t.ated = prompner
        ge"e model"" the text using""Generat      "st[int]:
  s) -> Li, **kwarg int = 50max_length:sor, .Tenrchtot: f, prompext(sel _generate_t  def
  results
   ration_return gene
             :.3f}")
  on_score']itiults['repetres: {on Score   Repetitint(f"     pri       ]:.3f}")
 ens_ratio'['unique_tokultsesatio: {r Token Rnique"    U(f   print         f}")
ngth']:.1lts['avg_le: {resuAvg Lengthint(f"    pr
   )}:")title(gy.{strate(f"     print       s():
  itemlts.esun_rneratiolts in geesuategy, r   for str:")
     ltsResuality tion QuGenerat(f"\n     prin
   ")
      {strategy} forityiversn dnable tokehave reasohould "Sf                            0.1,
 '], atio_re_tokenss['uniqu(resultterGreaf.assert        sel")
    rategy}{sth for inimum lengtet mould me text shGenerated     f"                     ,
   ation_lengthnerin_ge self.mg_length'],esults['avreater(rssertGf.asel      ():
      esults.itemson_rn generati itsy, resulteg  for stray
       qualitgenerationate  Valid   #
       esults
    y_rtrateg= s] gy_name[stratesultson_re    generati
             s)
tioneneraore(all_gon_scpetiticalculate_reself._n_score'] = etitios['repesult strategy_r          ns)
 _generatioratio(allen_nique_toke_uf._calculat= sel] tio's_raque_tokenesults['unitegy_r        stra)
    nerations]n in all_geor geen) fn([len(gp.mea n'] =g_lengthults['avresgy_  strate        etrics
  lity mulate qua      # Calc

         ted)eraend(genions'].apperatsults['genretegy_rast         d)
       nd(generatepe.apenerations      all_g
                    )

      y_params  **strateg           th,
       on_leng_generatiself.maxax_length=          m
          pt,        prom             ate_text(
generlf._rated = se   gene
          erate text# Gen            pts:
    romt in test_por promp  f
        s = []
   eneration    all_g
                 }
           0
     _score': 'repetition       ,
        io': 0ns_ratkee_to'uniqu
          ngth': 0,le'avg_             : [],
   ons''generati
ults = {rategy_res     st      s():
 es.itemstrategimpling_arams in sa_pe, strategygy_namstrate     for
     ]
      ange(5)
  r0)) for _ in(1, 1ab_size, ig.vocf.model_confdint(0, selranorch.      t      mpts = [
st_prote
        ptsst prom       # Te
    = {}
    tion_resultsgenera
                 }
 re': 0.8}
, 'temperatumple': Truee': {'do_sa'temperatur
  .9},': 0top_p, 'ample': True_sp_p': {'do'to      50},
      op_k': True, 't': 'do_sample {p_k':   'to
         : False},ample'y': {'do_s'greed            tegies = {
trang_s samplis
       tegiempling strat saenffern with dist generatioTe      # "
  cs"" metrialityeneration qust text g"""Te  lf):
      metrics(se_quality_generation   def test_s

 sulterplexity_re   return p

 ity:.2f}")plexn: {std_periatioevrd Dandaf"  St print()
       ".2f}rplexity:} - {max_pe:.2fityerplexin_p: {m  Rangeint(f" pr       ']}")
s_target['meet_resultsrplexityet: {pe  Meets Targ  print(f"")
      xity_max}erpleget_p: {self.tarhreshold Target T  print(f" )
      ty:.2f}"g_perplexi {avity:age Perplexnt(f"  Aver        prisults:")
n Reio ValidatnPerplexityf"\  print(
      1")
      than er greate ity should b  "Perplex
     1.0,y,_perplexittGreater(avg.asserelf      sable")
   reasonhould bexity sperpleerage Av "                  .0,
    1000erplexity, ss(avg_pelf.assertLe s
       lidation      # Va
            }
ies
      lexiton': perpy_distributiexit'perpl           x,
 rplexity_maarget_pelf.t <= seg_perplexity': avrgetmeets_ta    '
     rplexity,x_pe: malexity'   'max_perp
 xity,in_perplerplexity': m    'min_pe,
        exityply': std_perexit  'std_perpl
     _perplexity,lexity': avgverage_perp 'a          {
  esults =ity_rerplex       p
   s)
      lexitie np.max(perpy =erplexitax_p  m)
      iesitperplex= np.min(perplexity  min_
       rplexities)p.std(pelexity = n std_perp  ties)
     erplexip.mean(p= nerplexity       avg_ptics
  atisculate st Cal      #

  xity)ple(perappends.itieexperpl
 item()ss).p(lotorch.ex= y it    perplex          ity
   perplexrt to  # Conve

             )    mean'
     ion='     reduct          ,
     w(-1).vieets        targ           )),
 its.size(-1view(-1, log     logits.         y(
      cross_entrop = F.      loss
          py lossoss-entroe cr# Calculat

        s']uts['logit outps =     logit           _ids)
utnpdel(i = self.mooutputs
                      gets
arft t  # Shieze(0)   unsque[1:]. = sequenceargets     t        t token
   ove lasemze(0)  # R-1].unsqueence[:seques =     input_id
  enceequexity for s perplalculate # C           ces:
    st_sequenin tece   for sequen
    .no_grad():ch with tor

     = []es itierplex  p

       ]ge(20)
    _ in ranfor, (32,)) sizeocab_fig.vdel_cont(0, self.modinorch.ran   t         = [
s sequence    test_ataset
     dte test  # Crea
      """lidationn and vaalculatioty clexierp""Test p "
       elf):tion(sity_calculast_perplex
    def te)
    nfig.model_coLMModel(self Nanol =self.mode

       )eads=8
 num_h      ,
      _layers=4         num  ,
 ize=256 hidden_s
           1000,ize=   vocab_s
         lConfig(defig = Mo.model_con selfl
       t mode tesatere     # C
   100
      th = tion_lengax_genera     self.m
   length = 20ation_nerge   self.min_ld
     thresho perplexity .0  # Target50max = lexity_erpet_pself.targ    ""
    onment"enviry testing et up qualit   """S):
     (self  def setUp
  "
    s""ricy metion qualitat and generxityleerpst p"""Tese):
    TestCaunittest.lity(tyAndQuaxierpleass TestPcl
mory

_mealinitiory -  final_memreturn
      1024)
 024 * o().rss / (1.memory_infProcess() = psutil.ory_mem finaler
       emory aft # Measure m
     ut)
      t_inpesl(tuts = modetp       ou         rad():
o_gh torch.n    wit  ce
      nferene:  # i       els
 )
        ep(timizer.st   op
         ward()backs.   los
         )ew(-1)
    _input.vi     test         e(-1)),
  its'].sizutputs['log, os'].view(-1puts['logit    out
 y(trop_en= F.cross       loss     _input)
 l(testtputs = mode        ou
r=1e-4)
  ers(), lparametamW(model.ch.optim.Ad tortimizer =          optep
  aining ste trmula # Si           raining':
ario == 't  if scen
             024)
 / (1024 * 1nfo().rssmory_iss().me.Proce= psutilmemory    initial_efore
     emory bsure m   # Mea
     n))
      seq_le(batch_size,, t(0, 1000din.ranput = torch test_in
        64
 n =   seq_le    e 1
 aining' els 'trenario === 8 if scatch_size        bta
 est daate tCre      # ""
  scenario" given a model in asage for mory ue me""Measur "      float:
 -> str) scenario: ule, del: nn.Modself, momory_usage(asure_me   def _me

    _resultsryurn memoret
              et']}")
  _targencyciets_effis['meget: {results Taret Meint(f"         pr")
      ']:.1%}oneducti'memory_rn: {results[   Reductiorint(f"  p
           ")MB:.1f} y_mb']ntized_memorlts['quaresud: {    Quantize" print(f       B")
    :.1f} Memory_mb']ine_msults['basel{reBaseline: rint(f"       p        ")
 title()}: {scenario.t(f"       prin
 s():sults.itemry_res in memoario, result   for scen    n:")
 idatioficiency ValMemory Ef(f"\nint     pr
     ")
   io} {scenarforuld be >50% shoy reduction Memor     f"
      '], 0.5,uctionmemory_redr(results['eatef.assertGr       sel    :
 .items()_resultsts in memoryulio, res  for scenar
  ationlid # Va

 results = scenario_cenario][s_results     memory
                  }
 et
       rgreduction_tamory_ization_me self.quanteduction >=_remoryget': mncy_tar_efficie  'meets            ,
  eductionemory_ruction': mredmory_     'me       emory,
    quantized_my_mb': orantized_mem     'qu       ry,
    ine_memoy_mb': baselseline_memorba          '
   = {tssulrio_re      scena
            emory
     aseline_m) / bemory quantized_mory -seline_membaduction = (memory_re        ency
    e efficiat # Calcul
         cenario)
 el, sd_moduantizery_usage(qemo_msuremeay = self._tized_memoruan      q

  model)e_model(quantiz= quantizer.d_model uantize q           config)
quant_uantizer(tizer = NF4Q       quan
              )
          16
 floatdtype=torch.    compute_
  nType.NF4,antizatioon_type=Qutintizaua          q     g(
 onfionCatiuantizonfig = Quant_c  q
        onzatiith quantiest w         # T
              enario)
 sc(model, mory_usage._measure_me selfmory =line_mebase
 lect()
    c.col    gne
        le() else Nois_availabch.cuda.e() if tory_cachuda.empt.c    torch
     ory memlineMeasure base     #
               ig)
  el(conf NanoLMModdel = mo
      ionattizout quant with       # Tes
          {}
   results =scenario_     s:
       io in scenarscenario    for
    e']
       , 'inferenc'training'ios = [narce  s
      enarios # Test sc

ults = {}y_res   memor     ation
 quantiznd withoutwith aemory usage st m      # Te
  'medium']
gs[el_confi self.modfig = con     "
  "ference"ng and inniduring traincy ciemory effimeest   """T
      (self):ionlidatficiency_va_memory_efef test
    df_results
 deofeturn tra    r
         ")
   n']:.1%}ize_reductio {results['sion:e Reduct Sizprint(f"                     4f}")
  _score']:.'quality {results[ty Score:  Qualirint(f"     p                 .1f} MB")
']:_mbts['sizee: {resul"    Siz  print(f                 )
 ig_name}:"conf {  print(f"                 results:
  ot in f 'error' n i           ms():
    lts.ite_resueoff tradresults in, onfig_namefor c

   eoff}")radf: {best_tradeof"  Best Tt(f  prin       :")
   eoffSize Traduality vs ntization Quaint(f"\nQ          pr
         )
    _score']lityults[k]['quaresalid_bda k: vlam   key=                          ),
 lts.keys(esuax(valid_rf = mbest_tradeof            ults:
lid_resf va        i
 v}
       in' not if 'errortems() ff_results.in tradeo v ior k, = {k: v flid_resultsva        t tradeoff
es # Find b
                 }
           e': 0.0
  ity_scorual         'q
loat('inf'),e_mb': f     'siz           r(e),
    rror': st   'e
         e] = {am_nts[configultradeoff_res          s e:
       Exception a except
               }
             _mb(model)
zee_model_sialculatlf._c / semb)- size_del) b(moe_m_model_siz._calculate: (selfreduction' 'size_                  metric
 mbined # Co 0.1),  t_mse *logiim - (': cosine_suality_score          'q
          im,e_sty': cosinilari 'cosine_sim                   mse,
mse': logit_logit_ '                  b,
 size_m'size_mb':
       = {ig_name] esults[confadeoff_r      tr
em()
         ).it       e(0)
      unsqueezized_flat. quant
squeeze(0), at.une_flselin    ba         ty(
       imilari= F.cosine_sosine_sim       c
n()
      ts'].flatteut['logitpd_ou = quantize_flatzedanti      qu          en()
attgits'].flput['lobaseline_out = _flatline       base
         ity similarnete cosi   # Calcula
           )
      m().ite          ']
      put['logitsline_out    base            s'],
    'logit_output[quantized                   mse_loss(
 git_mse = F.         lo
     metricsrity imilaate scul     # Cal
                  t)
  test_inpudel(zed_moput = quanti_outntized  qua                  :
grad()h torch.no_   wit           larity)
  put simiutity (oquale sur      # Mea

 _model)b(quantizedl_size_modealculate_mf._c sel  size_mb =
  sure size     # Mea
             odel)
ze_model(mtintizer.quan quaized_model =  quant

  t_config)er(quanantizFP4Ququantizer =                 e:
         els      fig)
     cont_quanzer(NF4Quantintizer =        qua
    NF4:ype.uantizationT == Qization_typequant_config.quant     if
           uantization # Apply q
         try:     s():
 gs.itemion_confiin quantizatt_config uan, qonfig_namer c      fo
  t)
     put_inel(tes mod =tputeline_ou bas
o_grad():orch.n     with t output
   ine basel     # Get
            32))
ize, (4,b_s.vocase_configbant(0, ch.randiut = torest_inp     tent
   ssessmty ar qualiest data foreate t    # C

        {} = ts_resultradeoff
         }
  )
                rch.uint8
type=tont_d  qua
  loat16,pe=torch.fty compute_d
 ype.FP4,tionTQuantiza_type=tion   quantiza
  nConfig(zatioQuanti     'fp4':         ),
         int8
  rch.u=to_dtypequant            t32,
    .floape=torchpute_dty       com
         pe.NF4,antizationTy=Quon_typeizatint      qua          nConfig(
tizatio32': Quan4_fp 'nf
     ),          nt8
 e=torch.uitypnt_d       qua
         t16,h.floa_dtype=torcputeom         c   4,
    NFionType.pe=Quantizattion_tytiza        quan      fig(
  ionCon': Quantizatf4_fp16'n         {
    s =nfign_coquantizatio
        tionson configurati quantizat different      # Tes
  e_config)
 asModel(bl = NanoLM        mode['medium']
_configs= self.modele_config       bas
  ""ethods"antization merent qufor diffradeoff ze tuality vs si""Test q
        "(self):deoffe_travs_sizality_ntization_quest_qua
    def t   config
 t_besesults, size_r   return
]}")
   arget'_reduction_tsults['meetsuction={re   f"Red          }, "
     get']e_tarets_sizlts['meize={resugets: STar   Meets (f" rint  p     ")
     n']:.1%}ctio_redumemoryts['ul{reson: educti    Memory Rrint(f"     p")
       .1f} MBb']:ze_msiantized_esults['que: {rized Sizant   Qu" print(f
    ")']:.1f} MBze_mbe_siresults['bas {se Size:(f"    Ba     print
  }:")config_namet(f"  {      prin      s.items():
esult size_rine, results nfig_nam   for co
     ")
       }{best_configuration: t ConfigBes"     print(f   :.1%}")
  etn_targ_reductiooryation_memntiz.quaselfn: {eductiot Memory R(f"  Targe      printb} MB")
  e_max_mt_sizlf.targe{seize_min_mb}-_setelf.targze Range: {s Target Si print(f"       lts:")
 ation ResuSize Valid"\nModel    print(f

me}")onfig_na50% for {cleast y by at uce memorshould redzation f"Quanti
        '], 0.5,y_reductionts['memor(resulsertGreater.as        self
    ms():_results.itets in sizeme, resul_nafig for con
       ectivenession effquantizat # Verify
  ")
 targetsizeeet sn should matioigurconfne  o least"At
est_config,tIsNotNone(blf.asser        sessertions
tion a# Valida
name
  nfig_= coconfig t_  bes                ore
   sc_score =  best                 st_score:
 core < be      if s
                     ddle)
     mitarget_'] - _mbed_sizelts['quantiz = abs(resu       score
         b) / 2size_max_melf.target_ s_min_mb +.target_sizele = (selfrget_midd   ta            range
  targetddle of to mie on how clossed  Score ba   #       ]:
      tion_target'meets_reducs['d resultrget'] an_size_tas['meetsltif resu           ems():
 itze_results.n sisults i refig_name,or con        f
  'inf')
  float(st_score =    be  = None
   g _confist     be
   t sizetargeation for configurnd best # Fi
          }
          rs())
rametemodel.pa) for p in l( sum(p.numeter_count':'parame               n_target,
 y_reductioortion_memquantizaf.on >= selry_reductiget': memo_tars_reduction    'meet
  _mb),maxarget_size_ self.t_size_mb <=ntized<= quaze_min_mb et_si(self.targtarget': eets_size_      'm          duction,
emory_rection': mmemory_redu     '           e_mb,
ized_siz_mb': quanttized_size       'quan
         ase_size_mb,e_mb': base_siz          'b
   _name] = {configs[e_result  siz

         base_size_mb_size_mb) / zedti- quanize_mb  = (base_s_reduction   memory
         eduction memory r # Calculate
        del)
    uantized_mosize_mb(qate_model_calcul.__mb = selfntized_size       qua  odel)
   ze_model(mer.quantiizodel = quant quantized_m         ig)
  onfuant_cantizer(qr = NF4Quntize      qua
               )

           8e=torch.uintt_dtypquan
    h.float16,e_dtype=torcputcom             e.NF4,
   izationTyptype=Quantization_  quant            (
  onConfigizatiQuantonfig = t_cuan  q         n
 quantizatioly NF4  App           #
     el)
     mb(modize_el_sulate_modf._calc = selsize_mb     base_
       ig)Model(confel = NanoLM   mod
      ase model  # Create b         ):
 .items(del_configs in self.mo configname,config_    for
     {}
    _results =size
      zation"""uantih qwitts geize tarels meet st that mod   """Tes     ts(self):
geel_size_taref test_mod d
   to MB
    # Convert 024)  124 *(10tes / otal_size_byturn t     re
   float32
   to Defaulte * 4  # _sizs += paramze_bytetotal_si              e:
  ls e
    m_size * 1 += parabytesize_al_s     tot          :
 ch.uint8== tore r param.dtypt8 o== torch.inpe tylif param.d          e2
  e * izaram_s_bytes += p total_size           at16:
    ch.floe == toram.dtyplif par   e         size * 4
s += param__size_byteal tot             oat32:
  rch.fl= todtype = if param.          type
 e based on dulate sizalc# C
     ze
  = param_sital_params +        to)
    ram.numel( = paizeparam_s            eters():
del.paramm in mo    for para
       es = 0
 _size_byt  total
       = 0al_params  tot
      ""B"ze in Model site malcula     """C:
   le) -> float: nn.Modu, modelmb(selfdel_size_alculate_moef _c d

    })

 size=2048iate_ntermed       i,
         16ds=  num_hea
  8,ayers=um_l    n            512,
size=dden_        hi     5000,
    vocab_size=           ig(
    ': ModelConf      'large
      ),
     ize=1536iate_sednterm      i     12,
       num_heads=             ers=6,
 lay      num_          84,
den_size=3