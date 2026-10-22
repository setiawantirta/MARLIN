"""Per-compound CMNPD curation worker (RDKit). Used with multiprocessing.Pool."""
import os, sys
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, QED, rdMolDescriptors, AllChem, MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import RDConfig
RDLogger.DisableLog('rdApp.*')
sys.path.append(os.path.join(RDConfig.RDContribDir, 'NP_Score'))
import npscorer

ALLOWED = {'H', 'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Se', 'Br', 'I'}
MW_MIN, MW_MAX = 100.0, 1500.0
HEAVY_MIN = 6

_G = {}


def _init():
    if 'te' not in _G:
        te = rdMolStandardize.TautomerEnumerator()
        te.SetMaxTautomers(200)
        te.SetMaxTransforms(200)
        _G['te'] = te
        _G['un'] = rdMolStandardize.Uncharger()
        _G['lfc'] = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
        _G['np'] = npscorer.readNPModel()
    return _G


def curate(rec):
    """rec = (compound_id, smiles, molblock_or_None) -> dict"""
    cid, smi, mb = rec
    G = _init()
    out = {'COMPOUND_ID': cid, 'smiles_input': smi, 'status': 'ok', 'reject_reason': '',
           'parsed_from': 'smiles', 'had_multiple_fragments': False,
           'charge_neutralized': False, 'tautomer_changed': False}
    mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
    if mol is None and mb:
        mol = Chem.MolFromMolBlock(mb, sanitize=True)
        out['parsed_from'] = 'sdf_molblock'
    if mol is None and mb:
        try:
            mol = Chem.MolFromMolBlock(mb, sanitize=False)
            if mol is not None:
                Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^
                                 Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                out['parsed_from'] = 'sdf_molblock_partial'
        except Exception:
            mol = None
    if mol is None:
        out['status'] = 'rejected'
        out['reject_reason'] = 'parse_fail'
        return out
    try:
        mol = rdMolStandardize.Cleanup(mol)
        nfrag = len(Chem.GetMolFrags(mol))
        if nfrag > 1:
            out['had_multiple_fragments'] = True
            mol = G['lfc'].choose(mol)
        fc_before = Chem.GetFormalCharge(mol)
        mol = G['un'].uncharge(mol)
        out['charge_neutralized'] = (Chem.GetFormalCharge(mol) != fc_before)
        smi_pre = Chem.MolToSmiles(mol)
        mol_t = G['te'].Canonicalize(mol)
        if mol_t is not None:
            mol = mol_t
        out['tautomer_changed'] = (Chem.MolToSmiles(mol) != smi_pre)
        Chem.SanitizeMol(mol)
    except Exception as e:
        out['status'] = 'rejected'
        out['reject_reason'] = 'standardize_fail:%s' % type(e).__name__
        return out

    syms = {a.GetSymbol() for a in mol.GetAtoms()}
    bad = sorted(syms - ALLOWED)
    mw = Descriptors.MolWt(mol)
    heavy = mol.GetNumHeavyAtoms()
    out['smiles_std'] = Chem.MolToSmiles(mol)
    out['mw'] = mw
    out['heavy_atoms'] = heavy
    out['elements'] = ','.join(sorted(syms))
    if bad:
        out['status'] = 'rejected'
        out['reject_reason'] = 'disallowed_elements:' + ','.join(bad)
        return out
    if not (MW_MIN <= mw <= MW_MAX):
        out['status'] = 'rejected'
        out['reject_reason'] = 'mw_out_of_range:%.1f' % mw
        return out
    if heavy < HEAVY_MIN:
        out['status'] = 'rejected'
        out['reject_reason'] = 'too_few_heavy_atoms:%d' % heavy
        return out

    try:
        out['inchikey'] = Chem.MolToInchiKey(mol)
    except Exception:
        out['inchikey'] = ''
    out['inchikey_skeleton'] = out['inchikey'].split('-')[0] if out['inchikey'] else ''

    ri = mol.GetRingInfo()
    macro = max([len(r) for r in ri.AtomRings()], default=0)
    out.update({
        'exact_mass': Descriptors.ExactMolWt(mol),
        'clogp': Crippen.MolLogP(mol),
        'tpsa': rdMolDescriptors.CalcTPSA(mol),
        'hbd': rdMolDescriptors.CalcNumHBD(mol),
        'hba': rdMolDescriptors.CalcNumHBA(mol),
        'rotb': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'n_rings': ri.NumRings(),
        'n_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
        'n_aliphatic_rings': rdMolDescriptors.CalcNumAliphaticRings(mol),
        'fraction_csp3': rdMolDescriptors.CalcFractionCSP3(mol),
        'n_stereocentres': rdMolDescriptors.CalcNumAtomStereoCenters(mol),
        'n_unspec_stereocentres': rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol),
        'qed': QED.qed(mol),
        'bertz_ct': Descriptors.BertzCT(mol),
        'n_heteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
        'max_ring_size': macro,
        'is_macrocycle': bool(macro >= 12),
        'formula': rdMolDescriptors.CalcMolFormula(mol),
        'n_amide_bonds': rdMolDescriptors.CalcNumAmideBonds(mol),
        'n_lipinski_hba': rdMolDescriptors.CalcNumLipinskiHBA(mol),
        'n_lipinski_hbd': rdMolDescriptors.CalcNumLipinskiHBD(mol),
    })
    try:
        out['np_likeness'] = npscorer.scoreMol(mol, G['np'])
    except Exception:
        out['np_likeness'] = float('nan')
    try:
        sc = MurckoScaffold.GetScaffoldForMol(mol)
        out['murcko_scaffold'] = Chem.MolToSmiles(sc) if sc.GetNumAtoms() else ''
        out['murcko_generic'] = Chem.MolToSmiles(
            MurckoScaffold.MakeScaffoldGeneric(sc)) if sc.GetNumAtoms() else ''
    except Exception:
        out['murcko_scaffold'] = ''
        out['murcko_generic'] = ''
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    out['ecfp4_onbits'] = list(fp.GetOnBits())
    out['maccs_onbits'] = list(MACCSkeys.GenMACCSKeys(mol).GetOnBits())
    return out
