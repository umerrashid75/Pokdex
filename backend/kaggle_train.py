#!/usr/bin/env python
# coding: utf-8

# =============================================================
# Smart Pokédex – Kaggle Training Notebook
# =============================================================
# Phase 1 : Fine-tune ResNet50  on 90 animal species
# Phase 2 : Fine-tune GPT-2     for Pokédex-style descriptions
#
# ▶ DATASETS TO ADD IN KAGGLE (+ Add Data button):
#   1. "Animal Image Dataset - 90 Different Animals"
#      kaggle datasets add iamsouravbanerjee/animal-image-dataset-90-different-animals
#
# ▶ OUTPUTS  →  /kaggle/working/
#   pokedex_classifier.pth   ← ResNet50 fine-tuned weights
#   pokedex_gpt2/            ← GPT-2 fine-tuned model folder
#   class_labels.json        ← {index: "animal_name"} for backend
# =============================================================

# Install extra deps (GPT-2 fine-tuning)
import subprocess
subprocess.run(["pip", "install", "-q", "transformers", "accelerate", "datasets"], check=True)

import os, json, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from torchvision import models
from torchvision.datasets import ImageFolder

print(f"PyTorch {torch.__version__}  |  CUDA: {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================
# ██████  PHASE 1 – RESNET-50 FINE-TUNING
# =============================================================

# ── Dataset path after adding on Kaggle ───────────────────────
ANIMALS_DIR = Path("/kaggle/input/animal-image-dataset-90-different-animals/animals/animals")
assert ANIMALS_DIR.exists(), f"Dataset not found at {ANIMALS_DIR}. Did you add the dataset?"

classes     = sorted([d.name for d in ANIMALS_DIR.iterdir() if d.is_dir()])
NUM_CLASSES = len(classes)
idx_to_class = {i: c for i, c in enumerate(classes)}
print(f"{NUM_CLASSES} classes: {classes[:8]} ...")

with open("/kaggle/working/class_labels.json", "w") as f:
    json.dump(idx_to_class, f, indent=2)

# ── Transforms ───────────────────────────────────────────────
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = T.Compose([
    T.RandomResizedCrop(224, scale=(0.65, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    T.RandomRotation(20),
    T.RandomGrayscale(p=0.05),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
val_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])

# ── Load & split 80/20 ───────────────────────────────────────
full_ds = ImageFolder(ANIMALS_DIR, transform=train_tf)
n_train = int(0.80 * len(full_ds))
n_val   = len(full_ds) - n_train
train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                generator=torch.Generator().manual_seed(42))

# Swap val transform
val_ds.dataset = ImageFolder(ANIMALS_DIR, transform=val_tf)

train_ld = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=4, pin_memory=True)
val_ld   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

# ── Model: ResNet50 with frozen early layers ──────────────────
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

for name, p in model.named_parameters():          # freeze layers 1-2
    if not any(k in name for k in ["layer3", "layer4", "fc"]):
        p.requires_grad = False

model.fc = nn.Sequential(
    nn.Dropout(0.45),
    nn.Linear(model.fc.in_features, NUM_CLASSES),
)
model = model.to(DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable:,}")

# ── Optimizer & Scheduler ────────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW([
    {"params": model.layer3.parameters(), "lr": 1e-5},
    {"params": model.layer4.parameters(), "lr": 3e-5},
    {"params": model.fc.parameters(),     "lr": 1e-4},
], weight_decay=1e-4)

EPOCHS    = 25
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=[1e-5, 3e-5, 1e-4],
    steps_per_epoch=len(train_ld), epochs=EPOCHS,
)

# ── Training Loop ────────────────────────────────────────────
best_acc  = 0.0
save_path = "/kaggle/working/pokedex_classifier.pth"

for ep in range(1, EPOCHS + 1):
    # Train
    model.train()
    t_loss = t_correct = t_total = 0
    for imgs, lbl in train_ld:
        imgs, lbl = imgs.to(DEVICE), lbl.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, lbl)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()
        t_loss    += loss.item() * imgs.size(0)
        t_correct += (out.argmax(1) == lbl).sum().item()
        t_total   += imgs.size(0)

    # Validate
    model.eval()
    v_correct = v_total = 0
    with torch.no_grad():
        for imgs, lbl in val_ld:
            imgs, lbl = imgs.to(DEVICE), lbl.to(DEVICE)
            v_correct += (model(imgs).argmax(1) == lbl).sum().item()
            v_total   += imgs.size(0)

    t_acc = t_correct / t_total
    v_acc = v_correct / v_total
    print(f"[{ep:02d}/{EPOCHS}]  loss={t_loss/t_total:.4f}  "
          f"train={t_acc*100:.1f}%  val={v_acc*100:.1f}%")

    if v_acc > best_acc:
        best_acc = v_acc
        torch.save({
            "epoch":        ep,
            "model_state":  model.state_dict(),
            "num_classes":  NUM_CLASSES,
            "class_labels": idx_to_class,
            "val_acc":      v_acc,
        }, save_path)
        print(f"  ✅ Saved best ({v_acc*100:.1f}%)")

print(f"\n🎉 Phase 1 done. Best val: {best_acc*100:.1f}%  →  {save_path}")


# =============================================================
# ██████  PHASE 2 – GPT-2 POKÉDEX DESCRIPTION GENERATION
# =============================================================
# We fine-tune GPT-2-small on a curated corpus of Pokédex-style
# lore entries covering the 90 animals in the dataset.
# Format per entry:
#   <ANIMAL>Lion</ANIMAL><ENTRY>A formidable apex predator…</ENTRY>
#
# At inference time the backend:
#   1. Feeds "<ANIMAL>GoldenEagle</ANIMAL><ENTRY>" as a prompt
#   2. GPT-2 auto-completes the Pokédex lore text
# =============================================================

from transformers import (
    GPT2LMHeadModel, GPT2TokenizerFast,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset as HFDataset

# ── 2.1 Training Corpus ──────────────────────────────────────
# Each entry is one training example.  Add/expand as you like.
RAW_ENTRIES = [
    ("Antelope", "A swift and vigilant grazer of open savanna. Its hollow horns are never shed—growing with the animal across its lifetime. When threatened, it emits a sharp bark and launches into a bounding sprint that can sustain speeds of 90 km/h."),
    ("Badger", "A fierce and industrious burrower whose underground setts span hundreds of meters of tunnel. Despite its stocky frame, it is one of the most aggressive animals for its size, capable of driving off animals many times larger."),
    ("Bat", "The only true flying mammal. It navigates the darkness using ultrasonic pulses, building a three-dimensional map of its environment from returning echoes. A single brown bat can consume 1,200 insects in a single hour."),
    ("Bear", "A massive omnivore with an olfactory system seven times more powerful than a bloodhound's. Before winter it enters a hyperphagia phase, consuming up to 20,000 calories daily to build the fatty reserves that sustain months of torpor."),
    ("Bee", "A eusocial insect whose colonies function as a superorganism. Worker bees communicate the precise location of flower patches through a waggle dance encoding both direction and distance relative to the sun."),
    ("Beetle", "The most species-rich order of animal life on Earth. Its elytra—hardened forewings—protect delicate membranous wings beneath while providing armoured defence. Some species can lift 850 times their own body weight."),
    ("Bison", "The heaviest land animal in North America. Once numbering sixty million across the Great Plains, its thundering herds shaped the entire prairie ecosystem. Despite its bulk, it can sprint at 65 km/h and leap fences taller than a human."),
    ("Boar", "A powerful and highly adaptable wild pig equipped with continuously growing tusks used for rooting, combat, and defence. Its sense of smell is so acute it is trained to locate truffles buried 20 cm underground."),
    ("Butterfly", "A master of metamorphosis, spending weeks dissolved into cellular soup inside its chrysalis before rebuilding into an entirely different form. Its wing scales are structured to diffract light, producing iridescent colours through physics rather than pigment."),
    ("Cat", "A solitary hypercarnivore whose retractable claws remain razor-sharp for ambush. Its pupils dilate to near-circular in low light, amplified by a tapetum lucidum that reflects photons back through the retina a second time."),
    ("Caterpillar", "The larval stage of Lepidoptera, consuming hundreds of times its own body weight in leaf matter before initiating the most dramatic biological transformation in the animal kingdom. Some species sequester plant toxins into their bodies as a chemical shield."),
    ("Chimpanzee", "Humanity's closest living relative, sharing over 98% of our DNA. It fashions and uses tools—stripping leaves from twigs to fish termites, and shaping stones as projectiles. Its social politics involve long-term alliances and calculated betrayals."),
    ("Cockroach", "A survivor of half a billion years, virtually unchanged since before the age of dinosaurs. It can live a week without its head, survive radiation doses lethal to most life forms, and hold its breath for 40 minutes underwater."),
    ("Cow", "A ruminant with four stomach chambers that allows it to extract nutrition from grass through fermentation. A single animal produces several hundred litres of methane daily as a byproduct of this microbial digestion."),
    ("Coyote", "A supremely adaptable canid that has expanded its range as wolves were eliminated, now inhabiting urban environments across North America. It communicates through a complex vocabulary of howls, yips, and barks audible for kilometers."),
    ("Crab", "A crustacean that walks sideways due to the articulation of its leg joints. Many species engage in mass migrations of millions of individuals to reach spawning grounds, halting traffic and temporarily closing roads in their wake."),
    ("Crow", "Among the most cognitively advanced birds ever studied. It recognises individual human faces, holds grudges across years, crafts multi-step tools, and has demonstrated the ability to plan for the future—a trait once considered exclusive to humans."),
    ("Deer", "A ruminant whose antlers are the fastest-growing tissue in the animal kingdom, adding up to 2.5 cm per day during the velvet phase. They are shed and regrown annually, with each generation larger than the last."),
    ("Dog", "The first animal domesticated by humans, a partnership forged at least 15,000 years ago. Its olfactory epithelium contains 300 million scent receptors—compared to 6 million in humans—making it a living chemical sensor of extraordinary precision."),
    ("Dolphin", "A highly social cetacean that sleeps with one brain hemisphere at a time, allowing constant awareness and uninterrupted swimming. It uses a signature whistle—equivalent to a name—to identify itself to others across long ocean distances."),
    ("Donkey", "A remarkably sturdy equid with exceptional endurance in extreme heat and drought. It has wide, flat hooves evolved for rocky terrain and can carry loads proportionally greater than a horse. Known for genuine willfulness rather than stubbornness."),
    ("Dragonfly", "The most successful aerial hunter alive, catching prey on over 95% of attacks—a success rate unmatched in the animal kingdom. Each of its four wings is independently controlled, enabling it to fly backwards, hover, and change direction instantaneously."),
    ("Duck", "A waterfowl with waterproof feathers maintained by a preen gland oil that creates a hydrophobic barrier. It can sleep with one eye open, activating only half its brain at a time to maintain vigilance while resting."),
    ("Eagle", "An apex avian predator with vision eight times sharper than a human's. Its brow ridge casts a shadow over its eyes, functioning as a natural sun visor. Talons generate a grip force of over 400 psi—enough to pierce bone."),
    ("Elephant", "The largest land animal alive. Its trunk contains 40,000 distinct muscle fascicles offering the dexterity to pick a single coin off a flat surface. It mourns its dead and has demonstrated self-recognition in mirrors—a rare cognitive milestone."),
    ("Flamingo", "A filter feeder that feeds with its head upside down, pumping water through a specialised keratinous sieve. Its pink colour is entirely dietary—derived from carotenoid pigments in the algae and crustaceans it consumes."),
    ("Fly", "An insect whose compound eyes provide nearly 360-degree vision and can detect up to 250 frames per second—compared to 60 for humans. It tastes with its feet through chemoreceptors on its tarsi, sampling food the instant it lands."),
    ("Fox", "A cunning canid that uses Earth's magnetic field as a targeting system when pouncing on prey buried under snow. Its bushy tail—called a brush—serves as both a rudder during high-speed turns and a warm wrap when sleeping."),
    ("Goat", "A supremely sure-footed climber whose hooves have hard outer edges for grip and soft inner pads that act like suction cups. Mountain goats can stand on ledges the width of a human hand on near-vertical cliff faces."),
    ("Goldfish", "A domesticated carp capable of recognising its keeper's face, navigating mazes, and remembering locations for months. The '10-second memory' claim is entirely false; studies show retention of learned tasks across years."),
    ("Goose", "A highly territorial and fiercely loyal waterfowl that mates for life. It will charge, strike with its powerful wings, and bite to defend its nest or partner from threats many times its own size."),
    ("Gorilla", "The largest living primate. Despite its fearsome appearance, it is predominantly gentle and herbivorous. It communicates through a vocabulary of over 25 distinct vocalisations and builds a fresh sleeping nest from branches every single night."),
    ("Grasshopper", "An orthopteran capable of jumping 20 times the length of its own body by storing elastic energy in a protein called resilin. In locust phase, a single swarm can consume 20,000 tonnes of crops in a single day."),
    ("Hamster", "A small rodent with cheek pouches capable of expanding to three times the size of its head, used to transport food to its burrow at remarkable efficiency. It is not naturally nocturnal but crepuscular—most active at dawn and dusk."),
    ("Hare", "A lagomorph capable of reaching 70 km/h in short bursts, using a galloping gait where its hind feet land ahead of its front feet. Unlike rabbits, its young are born fully furred and with open eyes—ready to run within hours."),
    ("Hedgehog", "A nocturnal insectivore protected by up to 7,000 hollow keratin spines. When threatened, it rolls into a perfect sphere. It is one of the few animals with natural resistance to many snake venoms."),
    ("Hippopotamus", "A semi-aquatic megafauna that secretes a red oily fluid often mistaken for blood—it is actually a natural sunscreen and antimicrobial agent. Despite its bulk, it is one of the most dangerous animals in Africa."),
    ("Hornbill", "A bird with a distinctive casque— a hollow protrusion atop its bill that functions as a resonating chamber to amplify its calls across dense forest. The female seals herself inside a tree cavity during nesting, fed through a small slit."),
    ("Horse", "A cursorial mammal whose leg anatomy converts tendons into springs, storing and releasing energy with each stride for extraordinary efficiency. It can sleep standing through a passive stay apparatus locking its limbs—lying down only for deep sleep."),
    ("Hummingbird", "The only bird capable of sustained hovering and backward flight. Its heart beats up to 1,260 times per minute during exertion. To survive cold nights, it enters a hibernation-like torpor, dropping its metabolic rate by 95%."),
    ("Hyena", "A highly intelligent social carnivore whose jaws generate bite forces among the highest of any land predator, capable of crushing bone that other scavengers cannot access. Contrary to reputation, it is a skilled hunter, not merely a scavenger."),
    ("Jellyfish", "An animal without a brain, heart, or bones that has survived five mass extinction events. One species—Turritopsis dohrnii—is considered biologically immortal, reverting back to its juvenile polyp stage upon reaching maturity."),
    ("Kangaroo", "A marsupial whose joey is born 33 days after conception, no larger than a jellybean. It completes its development inside the pouch over nine months. Adult males box rival males using forelimb strikes and powerful balancing kicks."),
    ("Koala", "A marsupial whose diet of eucalyptus leaves is so toxic and low in nutrition that it sleeps up to 22 hours per day to conserve energy for detoxification. Its fingerprints are virtually indistinguishable from a human's."),
    ("Ladybugs", "A beetle whose bright colouring is an honest warning signal to predators: it secretes a foul-tasting alkaloid fluid from its leg joints when threatened. A single individual consumes up to 5,000 aphids across its lifetime."),
    ("Leopard", "The most widespread of the large cats, surviving in habitats from rainforest to desert across two continents. It hauls prey—sometimes heavier than itself—up into tree branches to protect its kills from scavengers below."),
    ("Lion", "The only truly social big cat, living in prides. The male's mane signals health and testosterone—darker manes indicate higher dominance. Its roar can be heard 8 kilometers away and is used to broadcast territorial boundaries at dusk."),
    ("Lizard", "A reptile that regulates its body temperature with extraordinary precision by micro-positioning itself relative to sunlight and shade throughout the day. Some species shed their tail voluntarily as a decoy while they escape predators."),
    ("Lobster", "A crustacean that does not appear to age in the conventional sense—it grows larger and more fertile with age rather than weaker. It can regenerate lost claws and communicate dominance through released chemical signals in its urine."),
    ("Mosquito", "The deadliest animal to humans in history—indirectly responsible for more deaths than all wars combined through malaria, dengue, and yellow fever transmission. The female locates hosts using CO2, body heat, and chemical signatures of sweat."),
    ("Moth", "A nocturnal lepidopteran that navigates by maintaining a fixed angle to the moon—an ancient and reliable compass that fails catastrophically near artificial light, causing the spiraling behaviour humans observe. Many species do not eat as adults."),
    ("Mouse", "A small rodent with a highly developed ultrasonic communication system—its vocalisations occur far above human hearing range. It carries a whisker map of its environment, each whisker connected to a dedicated region of its sensory cortex."),
    ("Octopus", "A cephalopod with three hearts, blue copper-based blood, and a distributed nervous system—two-thirds of its neurons are located in its eight arms. It can solve puzzles, recognise individual humans, and camouflage itself in under 200 milliseconds."),
    ("Okapi", "The only living relative of the giraffe, discovered by western science as recently as 1901 despite having been known to local populations for millennia. Its tongue is long enough to wash its own eyelids."),
    ("Orangutan", "The most solitary of the great apes, whose social structure centres on lone adult males maintaining territories across vast forest ranges. They construct a new roofed sleeping platform from bent branches every single night."),
    ("Otter", "A semi-aquatic mustelid with the densest fur of any mammal—up to one million hairs per square centimetre—trapping an insulating layer of air. Sea otters hold hands while sleeping to prevent drifting apart in currents."),
    ("Owl", "A nocturnal raptor whose facial disc is a parabolic sound collector directing even the faintest rustling under 50 cm of snow directly to asymmetrically-placed ears—allowing three-dimensional sound triangulation in total darkness."),
    ("Ox", "A domesticated bovine that has served as the primary engine of human agricultural civilisation for seven thousand years. Its patient, powerful temperament and tolerance for heavy yoke labour made the expansion of farming economies possible."),
    ("Oyster", "A bivalve mollusc and keystone species that filters up to 200 litres of water per day, removing nitrogen, sediment, and pathogens. It can change biological sex multiple times across its life based on environmental conditions."),
    ("Panda", "A bear adapted to a bamboo diet despite possessing the digestive system of a carnivore. It must consume up to 38 kg of bamboo per day just to meet its caloric needs. Its pseudo-thumb is an enlarged wrist bone, not a true digit."),
    ("Parrot", "A psittacine with a syrinx capable of replicating human speech with phonetic accuracy. Beyond mimicry, studies show genuine understanding of word meaning and referential use in some species. They form lifelong pair bonds and grieve lost partners."),
    ("Pelecan", "A colonial waterbird with a throat pouch capable of holding three times the volume of its stomach. It hunts co-operatively, forming crescent-shaped lines to drive fish into shallows before simultaneously scooping and draining water."),
    ("Penguin", "A flightless seabird that repurposed its wings into hydrodynamic flippers, reaching swimming speeds of 35 km/h. In the Antarctic winter, male Emperor penguins huddle in rotating formations to share body heat across months of polar darkness."),
    ("Pig", "Scientifically classified among the most intelligent mammals, with cognitive abilities comparable to a three-year-old human child. Its olfactory capability is used to locate truffles, narcotics, and even landmines buried underground."),
    ("Pigeon", "A bird with a magnetic mineral in its beak that allows it to detect Earth's magnetic field with compass accuracy. Homing pigeons have returned to their lofts across distances exceeding 1,800 km—navigating through mechanisms still not fully understood."),
    ("Porcupine", "A rodent whose 30,000 hollow quills are modified hairs coated in microscopic backward-facing barbs. When they penetrate skin, muscle contractions drive them deeper. It releases a warning odour before deploying its quills."),
    ("Possum", "A marsupial whose thanatosis—playing dead—is involuntary: an autonomic nervous response to extreme threat rather than a conscious strategy. It also emits a foul decay-mimicking odour during this state to reinforce the illusion."),
    ("Raccoon", "A highly dexterous mammal with forepaws containing four times the sensory nerve density of human hands. It 'washes' food not to clean it but to enhance tactile sensation—wetness activates additional mechanoreceptors in its paws."),
    ("Rat", "A social rodent with a sophisticated emotional life: it laughs at ultrasonic frequencies when tickled, will sacrifice its own food reward to free a trapped companion, and exhibits regret-like brain activity after making suboptimal choices."),
    ("Reindeer", "The only deer species in which both sexes grow antlers. Its eyes shift from gold in summer to blue in winter—changing the reflective properties of the tapetum lucidum to enhance vision in the ultraviolet-rich Arctic twilight."),
    ("Rhinoceros", "A megafauna whose horn is composed entirely of compressed keratin—the same protein as human fingernails. Despite its armoured appearance, its skin is sensitive to sunburn and parasites, leading it to wallow in mud as natural protection."),
    ("Sandpiper", "A migratory shorebird that completes one of the longest non-stop flights in the world—the Bar-tailed Godpiper flies 11,000 km without landing, feeding, or drinking. It shrinks its digestive organs before migration to reduce weight."),
    ("Seahorse", "The only species on Earth in which the male gestates and gives birth to young. Its armoured body is made of bony plates rather than scales, and it propels itself using a dorsal fin that beats up to 70 times per second."),
    ("Seal", "A pinniped whose blubber layer can reach 10 cm thick, providing both thermal insulation and an energy reserve for months at sea. It slows its heart to as few as 4 beats per minute during deep dives through the mammalian dive reflex."),
    ("Shark", "A cartilaginous apex predator with electroreceptors called ampullae of Lorenzini that detect the bioelectric fields generated by a fish's beating heart from 50 metres away. Its skin is covered in dermal denticles—microscopic teeth that reduce drag."),
    ("Sheep", "A highly social ruminant that can recognise and remember up to 50 individual sheep faces for two years, as well as human faces. It uses the same region of its brain to process faces as humans do—a convergent cognitive evolution."),
    ("Snake", "A limbless reptile that 'sees' heat through pit organs capable of detecting temperature differentials of 0.003°C, constructing a thermal image of its environment overlaid with its visual field. It swallows prey whole through a highly flexible jaw."),
    ("Sparrow", "A small passerine with a complex vocal learning system similar to that of humans—juveniles learn songs by imitation, develop regional dialects, and can construct novel song sequences not present in their tutor's repertoire."),
    ("Squid", "A cephalopod that can change not only colour but skin texture within milliseconds using chromatophores, iridophores, and papillae under direct nervous control—allowing it to vanish against virtually any background."),
    ("Squirrel", "A rodent whose spatial memory allows it to relocate thousands of individually cached food items across a territory. It also practices 'deceptive caching'—pretending to bury food in the presence of rivals to protect its real stores."),
    ("Starfish", "An echinoderm with no brain or blood that uses seawater pumped through its hydraulic vascular system to operate hundreds of tube feet. It digests prey externally by extruding its own stomach out through its mouth."),
    ("Swan", "A large waterfowl that forms monogamous pair bonds lasting decades. It swims with its young—called cygnets—riding on its back for protection. The mute swan is not silent: it produces a range of hissing, grunting, and bugling sounds."),
    ("Tiger", "The largest living cat. Each individual's stripe pattern is unique as a fingerprint. Unlike most cats, it actively seeks water—swimming rivers several kilometres wide to hunt and cool itself. Its roar can be heard three kilometres away."),
    ("Turkey", "A large gallinaceous bird capable of short explosive bursts of flight despite its bulk. Wild turkeys roost in trees and have a field of vision exceeding 270 degrees. Males display an elaborate fan of tail feathers during courtship."),
    ("Turtle", "A reptile whose shell is not an external object but an integral part of its skeleton—the spine and ribcage are fused to the upper shell. Some species navigate across thousands of kilometres of open ocean using Earth's magnetic field."),
    ("Whale", "The largest animal ever known to exist. The blue whale's heart weighs 180 kg and its arteries are wide enough for a human to crawl through. Its song can travel across entire ocean basins at frequencies below human hearing."),
    ("Wolf", "An apex social predator that operates within a precisely coordinated pack hierarchy. Its reintroduction into Yellowstone triggered a cascade of ecological changes known as a trophic cascade—reshaping rivers, forests, and even the behaviour of other animals."),
    ("Wombat", "A burrowing marsupial that produces cubic faeces—believed to aid in territorial marking by preventing waste from rolling away. Its backwards-facing pouch prevents dirt from entering the pouch during digging."),
    ("Woodpecker", "A bird whose skull contains a specialised spongy bone and hyoid bone wrapped around its brain like a seatbelt, absorbing the shock of 20 impacts per second at 25 km/h without causing neurological damage."),
    ("Zebra", "An equid whose black-and-white stripes are unique to each individual and serve to confuse insects—particularly biting flies, which struggle to land on high-contrast moving patterns. Within a herd, the stripes create a dazzling optical confusion for predators."),
]

# ── 2.2 Build & Save Corpus ──────────────────────────────────
CORPUS_PATH  = "/kaggle/working/pokedex_corpus.txt"
GPT2_OUT_DIR = "/kaggle/working/pokedex_gpt2"

with open(CORPUS_PATH, "w", encoding="utf-8") as f:
    for animal, entry in RAW_ENTRIES:
        line = f"<ANIMAL>{animal}</ANIMAL><ENTRY>{entry}</ENTRY>\n"
        f.writelines([line] * 6)    # repeat each entry 6× to boost rare classes

print(f"Corpus written: {CORPUS_PATH}  ({len(RAW_ENTRIES)} unique entries × 6)")

# ── 2.3 Tokenise for HuggingFace Trainer ─────────────────────
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

with open(CORPUS_PATH, encoding="utf-8") as f:
    raw_texts = [l.strip() for l in f if l.strip()]

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=256, padding="max_length")

hf_dataset = HFDataset.from_dict({"text": raw_texts})
tok_dataset = hf_dataset.map(tokenize, batched=True, remove_columns=["text"])

# ── 2.4 Fine-tune GPT-2-small ─────────────────────────────────
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,   # causal language modelling
)

training_args = TrainingArguments(
    output_dir             = GPT2_OUT_DIR,
    num_train_epochs       = 8,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 2,
    warmup_steps           = 100,
    weight_decay           = 0.01,
    learning_rate          = 5e-5,
    lr_scheduler_type      = "cosine",
    logging_steps          = 20,
    save_strategy          = "epoch",
    fp16                   = torch.cuda.is_available(),
    report_to              = "none",
    prediction_loss_only   = True,
)

trainer = Trainer(
    model           = gpt2_model,
    args            = training_args,
    train_dataset   = tok_dataset,
    data_collator   = data_collator,
)

print("\n🚀 Fine-tuning GPT-2 …")
trainer.train()
trainer.save_model(GPT2_OUT_DIR)
tokenizer.save_pretrained(GPT2_OUT_DIR)
print(f"✅ GPT-2 saved → {GPT2_OUT_DIR}")

# ── 2.5 Quick generation test ─────────────────────────────────
print("\n📖 Generation test:")
gpt2_model.eval()
for test_animal in ["Tiger", "Octopus", "Eagle"]:
    prompt  = f"<ANIMAL>{test_animal}</ANIMAL><ENTRY>"
    enc     = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = gpt2_model.generate(
            **enc,
            max_new_tokens    = 120,
            do_sample         = True,
            temperature       = 0.82,
            top_p             = 0.92,
            repetition_penalty= 1.15,
            pad_token_id      = tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    entry = text.split("<ENTRY>")[-1].split("</ENTRY>")[0].strip()
    print(f"\n  [{test_animal}]:  {entry[:200]} …")

# =============================================================
# ██████  DONE
# =============================================================
print("\n\n🎉 All done! Download these files from /kaggle/working/:")
print("  ├── pokedex_classifier.pth   ← drop into backend/")
print("  ├── class_labels.json        ← drop into backend/")
print("  └── pokedex_gpt2/            ← drop entire folder into backend/")

