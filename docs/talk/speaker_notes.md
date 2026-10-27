# Speaker notes: Big-model quality, small-model price

## 1. Big-model quality at a small-model price

Welcome. This talk is about getting the quality of a big model at the price of a small one.
A frontier model does specialist work well, but it needs a whole 8-GPU server. A small model is cheap, but it isn't good at the job. We used the big model to teach small models on three tasks: SQL, financial questions and competition maths. Then we served each student on a single TPU chip. Along the way we found that the recipe has to change when the student is already strong.
After that, Fred will show his experiment, which took a different route: teaching models to draw. ~1 min.

## 2. What we'll cover

The talk has five parts. First the problem. Then the three experiments and how we set them up. Then the training process, step by step. Then the results, including the point where the recipe had to change. And then Fred's experiment. At the end I'll compare our work with Meta's TorchTitan RL work, give you a recipe you can try yourself, and sum up. ~30 s.

## 3. The problem

Part 1: the problem. ~6 minutes.

## 4. Big models are expensive; small ones fall short

Let's start with the hardware. Our teacher, DeepSeek-V4.1-Flash, is a mixture-of-experts model with 552 billion parameters. One copy of it fills a whole server of eight H200 GPUs, which costs 36 dollars 60 an hour at the cheapest listed rate.
GPUs like these are hard to get. Each copy of the model ties up a full server, so serving more users means renting more servers. A single TPU v6e chip costs 2 dollars 70 an hour. The question is whether we can move useful work from the left side of this slide to the right.
We put a price on every correct answer. On the BIRD text-to-SQL benchmark, the teacher gets 60.7 percent of questions right, and a thousand correct answers cost a dollar 44. The untrained small model on one TPU chip costs 18 cents per thousand correct answers, but it gets less than a third of the questions right.
We don't want either of these. We want a model in the top-left corner, one that is cheap and accurate. Keep this chart in mind, because we'll come back to it. ~3 min.

## 5. Generic small models miss domain details

Here is what "weak in the domain" looks like, with a real BIRD question. The untrained small model escapes the apostrophe the way MySQL does, and SQLite rejects the query. After training, the student uses SQLite's doubled quote and gets the right rows.
On the right you can see that the untrained model solves 40 percent of the simple questions and only 10 percent of the hardest ones. 462 of its wrong answers aren't even valid SQL. This is the gap companies care about: their own schemas and their own SQL dialect.
Here was our bet. We use the big model as a teacher, train the small model on its answers, and serve the student on a single TPU chip.
Before we started, we wrote down what success would mean: the student had to be at least five points better than before, with a confidence interval above zero, and at least ten times cheaper per correct answer than the teacher. We tested the bet three times, and the third test showed us where it breaks down. ~2.5 min.

## 6. The experiments

Part 2: the experiment setup. ~5 minutes.

## 7. Each task has an automatic grader

We ran three experiments through the same pipeline. In BIRD, the model writes SQL against a database schema, and the grader runs the query and compares the rows. In FinQA, the model answers a numerical question about a page from an earnings report, which has text and a table, and the grader checks that the number is within one percent. In MATH, the model solves competition problems and puts its answer in a box, and the grader checks for an exact match.
Each task is one module in the code, with its own prompt, answer extraction and grader. Everything else is shared: generation, verification, training and serving. Every grader also passes two checks: gold answers score 100 percent and shuffled answers score close to zero.
We chose each student by how far it was behind the teacher, measured on the same benchmark with the same prompt. Qwen3-1.7B is about 30 points behind on both SQL and finance, so it has a lot to learn. The next size up, Qwen3-4B, already matched the teacher on BIRD, so there was nothing left to teach it.
For the third experiment we picked a different kind of student on purpose: Gemma 4 E2B on competition maths. It is already strong, at 74.9 percent, and much closer to the teacher. That let us test whether a teacher still helps a student like that. ~3 min.

## 8. Six decisions shaped the experiments

Six decisions shaped all three experiments. We picked tasks with automatic graders, so a single run takes a few minutes and can show a two-point change. We used a weak student and a strong one, to see when a teacher helps. We trained only on verified answers. We kept answers short, without long reasoning traces. We trained with LoRA on one TPU host. And we compared models only within the same run, alongside a control and a second copy of the baseline. ~1.5 min.

## 9. The journey

Part 3: the stack and the training journey. ~11 minutes.

## 10. Every experiment follows the same six steps

Here is the whole process on one slide. All three experiments follow it. We measure the gap and the noise. We generate training answers, from the teacher for a weak student or from the student itself for a strong one. We check every answer with the task's grader. We distil with LoRA on one TPU host. We reinforce with GRPO. And we serve each student on one chip and work out the price of each answer. Distillation takes 10 to 27 minutes, and the final students are 4.8 to 17.5 times cheaper per correct answer than the teacher.
Everything in the stack is open source. The teacher runs in Vertex AI Model Garden's vLLM container on eight H200s. The pipeline is plain Python standard-library scripts that write JSONL, so we can regenerate every number from the raw files. We serve students with vLLM-TPU, one copy per chip, and train them with Tunix and qwix LoRA in JAX on one v6e-8 host. The same trainer handles both Qwen3 and Gemma 4. The trainer, the server and storage all use one checkpoint format, Hugging Face safetensors. ~2 min.

## 11. Measure the noise, then choose the training answers

Step one is to check the measurements before running the experiment. Every grader passes two checks: gold answers score 100 percent and shuffled answers score almost zero. Then we serve two copies of each untrained model side by side. On FinQA the two copies differ by 1.9 points, because greedy decoding on TPU serving isn't perfectly repeatable. So every comparison in this talk comes from a single run that includes a second copy of the baseline. The gaps to the teacher are 30, 32 and 18 points.
Step two is choosing who writes the training answers. For a weak student, that's the teacher. We sample it four times per question at temperature 0.7, which gives 25 to 34 thousand answers per task in under 40 minutes. For a strong student, it's the student itself. We sample it eight times per question, and we don't need the teacher at all. We also train on the student's own answers as a control on every task, and on MATH that control won. Both teacher and student write the same short format: brief reasoning, then one answer. ~3 min.

## 12. Verify every answer, then distil on one TPU host

Step three is verification. Every answer goes through the task's grader, and we keep it only if it's correct. That threw out 40 percent of the teacher's BIRD answers and a quarter of its FinQA answers. The teacher is good, but it isn't perfect. On MATH we filter the student's own answers the same way. A separate audit re-grades every row we kept and checks for any overlap with the evaluation sets. It found no leaks on any of the three tasks.
Step four is distillation. We train low-rank adapters on all eight chips of one TPU host, with the same settings on every task. That's 117, 69 and 57 optimizer steps, and 10 to 27 minutes of training. Before trusting any result, we checked that the training model and the serving model agree: they pick the same top token 99.8 percent of the time on Qwen3 and 99.4 percent on Gemma. A full retrain also reproduced a byte-identical checkpoint. One lesson we learned the hard way: adapter weights have to stay in float32. ~3 min.

## 13. Reinforce with GRPO, then serve on a single chip

Step five is reinforcement learning. GRPO samples the student eight times per training question. An answer gets a reward of one if the grader accepts it, and the update pushes the student toward the better answers in each group. Only questions with mixed outcomes teach it anything. If all eight answers are right, or all eight are wrong, there's nothing to compare.
The bars sort every training question, in every round, into three groups. Teal means all eight answers were right, gold means some were right, and gray means all eight were wrong. On FinQA the teal block grows from 43 to 61 percent, and it grows by taking over the gold block. The gray block barely changes. So RL takes answers the student can sometimes reach and makes it give them every time. It doesn't teach the student questions it never solves; that's what distillation is for. On held-out data, RL is worth 4 points on BIRD, 5.8 on FinQA and 1.8 on MATH.
Step six is serving, and putting a price on each answer. Every model gets the same serving flags and one TPU chip per copy. We raise the load until the server starts returning errors, and we price the best rate it reached without errors. We measure cost per correct answer, because a fast wrong answer is worth nothing. On BIRD, a thousand correct answers from the trained student cost 9 cents. That's half the cost of the untrained model and a sixteenth of the teacher's. Answer length matters a lot here, as we'll see with MATH. ~3 min.

## 14. What we got

Part 4: results. ~11 minutes.

## 15. All three students improved

All three students improved, and every confidence interval is above zero. On BIRD, accuracy went from 30.6 to 52.2 percent, a gain of 21.6 points. On FinQA it went from 44 to 72.5 percent, a gain of 28.4 points, which is only three points short of the teacher. On MATH, Gemma went from 74.9 to 78.4 percent, a gain of 3.5 points. Look at the middle bar for MATH: it isn't trained on the teacher's data. On MATH, the best recipe trained the student on its own answers. The numbers under each chart split every gain into its steps.
Let's look at BIRD. On 1,534 questions over eleven databases the student never saw, accuracy went from 30.6 to 48.2 percent after distillation, and to 52.2 after GRPO. The teacher scores 60.7. Distillation gave the biggest gain, 17.6 points, and RL added four more. The final student closed 72 percent of the gap to the teacher, and producing it cost 136 dollars of compute.
FinQA was our replication. We used the same student, the same teacher, the same settings and the same six-round schedule, all fixed before the run. The only new code was the task module. On 1,147 test questions, accuracy went from 44 to 66.6 percent after distillation and to 72.5 after GRPO, 3.2 points short of the teacher. Both gains are far larger than the noise, and the final student closed 90 percent of the gap for 79 dollars. ~3 min.

## 16. The control experiment tells us which recipe to use

If you remember one slide from this talk, make it this one. In every experiment we trained a control: the same recipe on the same number of rows, but with the student's own verified answers instead of the teacher's. The chart shows the teacher's result minus the control's.
For the weak student, the teacher's answers clearly win, by 13 points on BIRD and 10 on FinQA. For the strong student it goes the other way: the teacher's answers score 4.3 points lower than the student's own. One extra SFT run told us which recipe each student needed.
Why did the teacher's answers hurt on MATH? Gemma already reasons well, and it writes long solutions. The teacher writes much shorter ones, about 600 tokens compared with Gemma's thousand. The distilled student copied that style and lost the most on the two hardest levels. It isn't running out of room, either: it hits the length limit less often than before.
We also tried teacher corrections, where the teacher sees the student's failed answer and writes a fix, and we train on the fixes. That was even worse, at minus 10.8 points. Training on the student's own verified answers gained about two points. ~3 min.

## 17. Counting cut-off answers as wrong made RL work

We had to make one more change before RL worked for the strong student. Gemma's most common failure is running out of room: about 15 percent of its answers hit the 2,048-token limit before giving a final answer. Our original GRPO threw those answers away, so the student's main failure never got a negative reward. We kept them, gave them a reward of zero, and trained only on the tokens the student actually wrote. That doubled the number of problems with mixed outcomes in each round, from about 600 to 1,300, and GRPO then added 1.82 points instead of 0.84. ~1.5 min.

## 18. The trained student is both cheap and accurate

Here's the chart from the start of the talk again. On BIRD, the trained student landed where we wanted it, in the cheap and accurate corner. It's 15.8 times cheaper per correct answer than the teacher. The table covers all three tasks. Hardware price gives a factor of 13.6 everywhere, and throughput and accuracy decide the rest. FinQA reaches 17.5 times. MATH reaches only 4.8 times, because Gemma's long answers make one chip slower than the teacher's eight GPUs. Short answers save real money.
Why should you believe these numbers? Every gain sits next to its own noise floor, which we measured in the same run with two copies of the untrained model. The SQL and finance gains are about ten times the noise. The MATH gain is small, 3.5 points, but its interval is well clear of the noise. Repeat evaluations agree within a point, the audit found no leaks, a retrain was byte-identical, and one script regenerates every number. ~3 min.

## 19. Fred teaches models to draw

Part 5: Fred's experiment. ~7 minutes.

## 20. The problem Fred set out to solve

Placeholder for Fred's experiment: the problem, why drawing, and what success means.

## 21. Fred's training journey

Placeholder: Fred's training steps in the same flow, his decisions, and what each step taught him.

## 22. What Fred found, and how his approach compares

Placeholder: Fred's results and the routes side by side.

## 23. Meta's TorchTitan RL work found similar lessons

How does this compare with other work? At the PyTorch Conference, Meta presented TorchTitan RL for long-horizon agents. They trained a Qwen3.5 9B model with RL on about a thousand cleaned terminal tasks, using eight B300s split into trainer, rollout and eval workers that run fully asynchronously. Their problem and scale are different. They go straight to RL on multi-turn agents, while we distil first and then run RL on single-turn answers.
But the lessons are similar. Both projects cleaned the data first: only 42 percent of their tasks were clean, and 40 percent of our teacher's answers were wrong. Both checked the numerics: a compiled kernel gave them a 2.5x log-prob mismatch, and we check that our trainer and server agree. And both had to reward failures carefully. They treat a dead sandbox, a missing submission and the context limit differently, and our MATH fix, counting cut-off answers as wrong, is the same idea. ~2 min.

## 24. What we learned

To wrap up: we used one pipeline on three tasks with two students. A 1.7-billion-parameter student gained 22 points on SQL and 28 on finance, and a strong Gemma student gained 3.5 points on competition maths. The students are 4.8 to 17.5 times cheaper per correct answer than the 552-billion-parameter teacher, and each one cost 79 to 159 dollars of compute to produce.
If your student is weak, distil the teacher. If it's strong, let it learn from its own answers. Then add RL, and make sure the student's main failure gets a negative reward. And measure carefully.
Here's the recipe, so you can try it on your own task. Measure the gap and the noise with one prompt. Generate two training sets, one from the teacher and one from the student itself. Keep only the answers your grader accepts, and check for leaks. Distil both sets with the same LoRA settings and keep the better one. That one comparison tells you whether your student is weak or strong. Then run GRPO with the grader as the reward, and count cut-off answers as wrong. Finally, serve on single chips and work out the price of a correct answer.
The code you write is one task module. It loads the questions, builds a versioned prompt, extracts the answer and grades it, with gold and shuffled checks. We added FinQA and MATH exactly this way.
Thank you; I'm happy to take questions. ~3 min.

## 25. Appendix

Appendix: backup slides for questions. Each slide names the main slide it backs.

## 26. 33 backup slides and the main slides they support

Appendix map: every backup slide and the main-deck slide it supports (arrow).

## 27. The teacher ties up a whole server, even when idle

Backup for slide 4 (One frontier model needs a whole GPU server). The teacher needs a whole 8-GPU server, takes about 25 minutes to start, and bills while idle.

## 28. How we price a correct answer

Backup for slide 4 (Every correct answer has a price). This is the formula behind every cost number, with the measured inputs. The pricing deliberately favours the teacher.

## 29. The prompt the student sees on BIRD

Backup for slide 5 (Generic small models miss domain details). The prompt is fixed and versioned. It holds the schema, example rows, the hint, the question and a format instruction.

## 30. How the FinQA task is set up

Backup for slide 7 (Each task has an automatic grader). This shows what the student sees on FinQA and how its answer is graded.

## 31. How the MATH task is set up

Backup for slide 7 (Each task has an automatic grader). We added the MATH task module the same way as FinQA.

## 32. A closer look at the two students

Backup for slide 7 (One student starts weak, the other starts strong). Both student families went through the same trainer.

## 33. We fixed the evaluation design before seeing results

Backup for slide 8 (Six decisions shaped the experiments). This is how we avoided tuning on the test set.

## 34. Answer length is a serving cost

Backup for slide 8 (Six decisions shaped the experiments). The cost advantage per correct answer depends on answer length.

## 35. Every grader passed two checks

Backup for slide 11 (Check the measurements before the experiment). Every grader passes two checks before we use it.

## 36. How much training data each task produced

Backup for slide 11 (Choose who writes the training answers). This shows how much training data each task produced and how much of it passed the grader.

## 37. The leak audit found nothing

Backup for slide 12 (Keep only the answers the grader accepts). Three layers keep evaluation data out of training: the prompt builder, the record builder with its audit, and the trainer.

## 38. From graded answers to training records

Backup for slide 12 (Keep only the answers the grader accepts). The loss covers only what the model should produce, and every record renders to exactly the prompt that vLLM serves.

## 39. Supervised fine-tuning settings

Backup for slide 12 (Distil the student on one TPU host). We used one SFT configuration for every task and both students.

## 40. The trainer and the server compute the same model

Backup for slide 12 (Distil the student on one TPU host). Before trusting any result, we checked that the training and serving models agree, and we ran an exact retrain.

## 41. What it took to add Gemma 4

Backup for slide 12 (Distil the student on one TPU host). This is what it took to add a second model family.

## 42. What happens in one GRPO round

Backup for slide 13 (RL makes sometimes-right answers always right). Each round serves the current student, samples, grades, keeps the questions with mixed outcomes, updates, merges and repeats.

## 43. RL stayed stable

Backup for slide 13 (RL makes sometimes-right answers always right). RL stayed close to the starting model. As the student improved, fewer questions had mixed outcomes.

## 44. What RL doesn't do

Backup for slide 13 (RL makes sometimes-right answers always right). These are the limits of the RL stage.

## 45. How we measured serving throughput

Backup for slide 13 (Each student runs on a single chip). This is how we measured throughput for the cost numbers.

## 46. What broke, and how we fixed it

Backup for slide 13 (Each student runs on a single chip). These are the operational problems we hit and how the pipeline handles them.

## 47. All BIRD comparisons and training-data options

Backup for slide 15 (All three students improved). Every BIRD comparison, with per-question win and loss counts, and the training-data options. CORR beat self-distillation by only 1.5 points [−1.0, +3.9].

## 48. All FinQA comparisons and the control

Backup for slide 15 (All three students improved). Every FinQA comparison, and why the control gains more here than on BIRD.

## 49. Every MATH result

Backup for slide 16 (A strong student lost accuracy copying the teacher). Every MATH result. The KD route doesn't catch up, even after GRPO.

## 50. Teacher corrections hurt on MATH

Backup for slide 16 (A strong student lost accuracy copying the teacher). Teacher corrections helped a weak student a little on BIRD and hurt a strong student badly on MATH.

## 51. How we score cut-off answers

Backup for slide 17 (Counting cut-off answers as wrong made RL work). This is how the cut-off fix works.

## 52. Cost per correct answer on each task

Backup for slide 18 (The trained student is both cheap and accurate). This breaks down the cost advantage on all three tasks.

## 53. How much compute each task used

Backup for slide 18 (The trained student is both cheap and accurate). These are TPU v6e chip-hours per stage. SFT is cheap, RL takes most of the compute, and MATH rollouts dominate because the answers are long. Utilisation was 3-5%, so these are upper bounds.

## 54. What these results don't show

Backup for slide 18 (Why we trust these results). These are the caveats we state up front.

## 55. Versions and machines

Backup for slide 10 (We built everything on open-source libraries). These are the exact versions and what each machine does.

## 56. How the code is organised

Backup for slide 10 (We built everything on open-source libraries). The two halves share data only through JSONL files and Hugging Face checkpoints.

## 57. Adding your own task or student

Backup for slide 24 (How to run this recipe on your own task). This is what you write to bring the recipe to a new task or model.

## 58. A closer look at TorchTitan RL

Backup for slide 23 (Meta's TorchTitan RL work found similar lessons). These are the design choices Meta described for TorchTitan RL on Terminal-Bench.

## 59. What we'd borrow from TorchTitan RL

Backup for slide 23 (Meta's TorchTitan RL work found similar lessons). The comparison suggests these next steps.
