from cellbin2.utils.common import FILES_TO_KEEP
from cellbin2.modules.naming import DumpPipelineFileNaming, DumpImageFileNaming, DumpMatrixFileNaming


def write_to_readme():
    import re
    from cellbin2.utils.common import TechType
    sn = 'A03599D1'
    save_dir = "/demo"
    im_type = TechType.DAPI
    im_type2 = TechType.IF
    m_type = TechType.Transcriptomics
    readme_p = "../../README.md"
    with open(readme_p, "r") as f:
        md_cont = f.read()
    pfn = DumpPipelineFileNaming(sn, save_dir=save_dir)
    ifn = DumpImageFileNaming(sn=sn, stain_type=im_type.name, save_dir=save_dir)
    ifn2 = DumpImageFileNaming(sn=sn, stain_type=im_type2.name, save_dir=save_dir)
    mfn = DumpMatrixFileNaming(sn=sn, m_type=m_type.name, save_dir=save_dir)
    all_ = [pfn, ifn, ifn2, mfn]
    table_md = "| File Name | Description |\n| ---- | ---- |\n"
    if_key_word = ["txt", "merged"]
    m_key_word = ["mask"]
    for n in all_:
        attrs = dir(n)
        for name in attrs:
            if name.startswith("__") or name.startswith("_") or not isinstance(n.__class__.__dict__.get(name),
                                                                               property):
                continue
            else:
                value = getattr(n, name)
                doc = getattr(n.__class__, name).__doc__
                value = value.name
                if isinstance(n, DumpImageFileNaming):
                    type_ = n.stain_type
                    if type_ not in [TechType.DAPI.name, TechType.ssDNA, TechType.HE.name] and \
                            any(kw in value for kw in if_key_word):
                        continue
                if isinstance(n, DumpMatrixFileNaming):
                    if any(kw in value for kw in m_key_word):
                        continue
                if isinstance(n, DumpImageFileNaming) or isinstance(n, DumpMatrixFileNaming):
                    if n.__class__.__dict__.get(name) not in FILES_TO_KEEP:
                        continue

                table_md += f"| {value} | {doc} |\n"

    print(table_md)
    updated_markdown_text = re.sub(r'# Outputs\n[\s\S]*?(?=\n#|\Z)', f'# Outputs\n\n{table_md}\n', md_cont)
    with open(readme_p, "w") as f:
        f.write(updated_markdown_text)


if __name__ == '__main__':
    write_to_readme()