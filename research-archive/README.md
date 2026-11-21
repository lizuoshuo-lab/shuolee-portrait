# Portrait reconstruction research backup

This branch preserves first-party source variants from static 3DGS and SMPL-X through native VHAP and scene Gaussian fusion. The repository root retains the existing stable SMPL entry point. Sources by experiment live under research-archive.

Unmodified SMPL source files are provided by the repository root at the parent commit. The variant folders store selected source differences. External Depth Anything 3 and PGSR checkouts are omitted. The decision index records available scalar statuses from experiment reports. A completed optimization and a quality approval are separate. The most recent shared-foreground check reproduced all 87 VHAP crop masks from the scene alpha. The 17,885 measured hair-edge samples differed by at most one native pixel. This verifies image-coordinate correspondence. The 00850 tuft still had an inward median mesh bias of 10.5 px after VHAP and a direct 301-vertex offset transfer did not fix it.

Person media, trained models, licensed model assets, credentials, workstation private key material and raw experiment logs are excluded from the remote backup. Reference footage and the assets required for training must be supplied separately.
