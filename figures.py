"""Manuscript figures from in-memory scores, atlas maps and aggregate estimates."""
from pathlib import Path
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.collections import PolyCollection
from matplotlib.ticker import MaxNLocator
import numpy as np

FIG = Path('figures')
BLUE, TEAL, ORANGE, INK = '#4b667f', '#008c8c', '#D55E00', '#223348'


def panel(ax, letter, text):
    ax.set_title(text, loc='left', fontsize=8, pad=9)
    ax.text(-.13, 1.04, letter, transform=ax.transAxes, weight='bold', fontsize=10)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf', 'png', 'svg'):
        kwargs = {'dpi': 300} if extension == 'png' else {}
        if extension in ('pdf', 'svg'):
            kwargs['metadata'] = {'CreationDate': None, 'ModDate': None} if extension == 'pdf' else {'Date': None}
        fig.savefig(FIG / f'{name}.{extension}', **kwargs)
    plt.close(fig)


def forest(ax, rows, labels, low='hc3_low', high='hc3_high', color=TEAL):
    for k, row in enumerate(rows):
        ax.plot([row[low], row[high]], [k, k], color=BLUE, lw=1.2)
        ax.scatter(row['beta'], k, color=color, s=19, zorder=3)
    ax.axvline(0, ls='--', color='.6', lw=.8)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels)-.5, -.5)


def boxes(ax, frame, column, group='isControl', labels=('Controls', 'TLE')):
    rng = np.random.default_rng(5660)
    for k, (value, color) in enumerate([(True, ORANGE), (False, TEAL)]):
        values = frame.loc[frame[group].eq(value), column].dropna().to_numpy()
        ax.boxplot(values, positions=[k], widths=.5, showfliers=False, patch_artist=True,
            boxprops=dict(facecolor=color, alpha=.17, edgecolor=color),
            medianprops=dict(color=INK), whiskerprops=dict(color=color), capprops=dict(color=color))
        x=k+rng.uniform(-.16,.16,len(values))
        ax.scatter(x, values, color=color, marker='o', s=9, alpha=.65, linewidths=0)
    ax.set_xticks([0, 1], labels)


def methods():
    """Figure 1: imaging and analysis workflow."""


    fig, ax = plt.subplots(figsize=(7.1, 2.8), layout='constrained')
    ax.set(xlim=(0, 14), ylim=(0, 4))
    ax.axis('off')
    labels = {
        'mri':(1.35,2.,'Structural and\ndiffusion MRI','82 brain regions\nThree centers'),
        'reference':(4.65,3.,'Control reference','Relative node strength\nObserved FA deviations'),
        'networks':(4.65,1.,'Participant networks','271 TLE / 205 controls\n110 outcome graphs'),
        'hubs':(8.65,3.,'Hub involvement','Spatially matched\nFA concentration'),
        'resilience':(8.65,1.,'Network resilience','Targeted vs random\nnode isolation'),
        'comparison':(12.4,2.,'Group comparisons','Site checks\nEngel outcomes')}
    texts = {}
    for key,(x,y,title,body) in labels.items():
        texts[key] = (ax.text(x,y,title,ha='left',va='top',fontsize=7.2,weight='bold',color='black',zorder=3),
                      ax.text(x,y,body,ha='left',va='top',fontsize=6.8,linespacing=1.35,color='black',zorder=3))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    padding = 8 * fig.dpi / 72
    gap = 5 * fig.dpi / 72
    inverse = ax.transData.inverted()
    items = {}
    for key,(title,body) in texts.items():
        tb,bb = title.get_window_extent(renderer),body.get_window_extent(renderer)
        width,height = max(tb.width,bb.width),tb.height+gap+bb.height
        cx,cy = ax.transData.transform(labels[key][:2])
        left,top = cx-width/2,cy+height/2
        title.set_position(inverse.transform((left,top)))
        body.set_position(inverse.transform((left,top-tb.height-gap)))
        (x,y),(right,upper) = inverse.transform([[left-padding,cy-height/2-padding],
                                               [left+width+padding,top+padding]])
        items[key]=(x,y,right-x,upper-y)
        color = '#4C72B0' if key=='comparison' else '#526477'
        ax.add_patch(FancyBboxPatch((x,y),right-x,upper-y,
            boxstyle='round,pad=0,rounding_size=.10',fc='#F7F9FC',ec=color,lw=1.1,zorder=2))

    def connect(source, target, source_height=.5, target_height=.5):
        sx,sy,sw,sh = items[source]
        tx,ty,tw,th = items[target]
        ax.add_patch(FancyArrowPatch((sx+sw,sy+sh*source_height),(tx,ty+th*target_height),
            arrowstyle='-|>',mutation_scale=10,shrinkA=3,shrinkB=3,
            lw=1,color='#526477',zorder=1))
    connect('mri','reference',.72,.5)
    connect('mri','networks',.28,.5)

    connect('reference','hubs',.72,.72)
    connect('reference','resilience',.28,.72)
    connect('networks','hubs',.72,.28)
    connect('networks','resilience',.28,.28)
    connect('hubs','comparison',.5,.75)
    connect('resilience','comparison',.5,.25)
    save(fig,'fig1_methods')


def cortical_surface_image(maps):
    """Render yabplot's Conte69/aparc surfaces without an OpenGL context."""
    from yabplot.data import get_surface_paths, _resolve_resource_path, _find_cortical_files, get_atlas_regions
    from yabplot.utils import load_gii, parse_lut
    from yabplot.mesh import map_values_to_surface
    cortical = {('L_' if row.roi.startswith('ctx-lh-') else 'R_') + row.roi[7:]: row.full_hubness
                for row in maps.itertuples() if row.roi.startswith('ctx-')}
    assert len(cortical) == 68
    assert set(cortical) <= set(get_atlas_regions('aparc', 'cortical'))
    limit = maps.full_hubness.abs().max()
    meshes = [load_gii(path) for path in get_surface_paths('midthickness', 'bmesh')]
    csv_path, lut_path = _find_cortical_files(_resolve_resource_path('aparc', 'cortical'), strict_name='aparc')
    labels = np.loadtxt(csv_path, dtype=int)
    ids, _, names, _ = parse_lut(lut_path)
    values = map_values_to_surface(cortical, labels, ids, names)
    assert len(values) == sum(len(v) for v, _ in meshes)
    split = len(meshes[0][0])
    hemi_values = [values[:split], values[split:]]
    surface, axes = plt.subplots(2,2,figsize=(5,4.1),layout='constrained')
    for ax, hemi, side, label in [(axes[0,0],0,-1,'L lateral'), (axes[0,1],1,1,'R lateral'),
                                  (axes[1,0],0,1,'L medial'), (axes[1,1],1,-1,'R medial')]:
        vertices, faces = meshes[hemi]
        triangles = vertices[faces]

        depth = triangles[:,:,0].mean(axis=1)*side
        projected = np.stack([triangles[:,:,1]*side,triangles[:,:,2]],axis=-1)
        face_values = hemi_values[hemi][faces].mean(axis=1)
        colors = plt.cm.coolwarm(plt.Normalize(-limit,limit)(face_values))
        colors[~np.isfinite(face_values)] = matplotlib.colors.to_rgba('#dedede')
        normals = np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])
        normals /= np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-12)
        light = np.array([side,.15,.35])
        light /= np.linalg.norm(light)
        shade = .65+.35*np.maximum(0,normals@light)
        colors[:,:3] *= shade[:,None]
        order = np.argsort(depth)
        ax.add_collection(PolyCollection(projected[order],facecolors=colors[order],
            edgecolors='none',antialiased=False,rasterized=True))
        ax.autoscale_view()
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(label,fontsize=11,pad=1)
    buffer = BytesIO()
    surface.savefig(buffer, format='png', dpi=400, bbox_inches='tight', pad_inches=.02)
    plt.close(surface)
    buffer.seek(0)
    return plt.imread(buffer)


def cohort_and_hubs(maps):
    """Figure 2: cohort counts, cortical hub map and strongest hubs."""
    surface_image = cortical_surface_image(maps)
    limit = maps.full_hubness.abs().max()
    fig = plt.figure(figsize=(7.1,4.3),layout='constrained')
    grid=fig.add_gridspec(2,2,height_ratios=[1.1,1.6],width_ratios=[1.,1.2])
    ax=fig.add_subplot(grid[0,0])
    ax.axis('off')
    cohort_ax=ax
    cards=[('Imaging pool','276 TLE + 216 controls'),
           ('Connected graphs','271 TLE + 205 controls'),
           ('Surgical outcomes','110 graph-eligible patients\n70 Engel I / 40 Engel II-IV')]
    card_text=[]
    for label,number in cards:
        title=ax.text(.03,.95,label,transform=ax.transAxes,fontsize=7.3,
                      weight='bold',color=INK,va='top',in_layout=False)
        body=ax.text(.03,.8,number,transform=ax.transAxes,fontsize=6.8,
                     color=INK,va='top',linespacing=1.25,in_layout=False)
        card_text.append((title,body))
    panel(ax,'A','Analysis cohorts')
    ax=fig.add_subplot(grid[:,1])
    ax.imshow(surface_image)
    ax.axis('off')
    color_scale=plt.cm.ScalarMappable(norm=plt.Normalize(-limit,limit),cmap='coolwarm')
    fig.colorbar(color_scale,ax=ax,orientation='horizontal',shrink=.85,pad=.02,
        label='Relative-strength hubness (SD)')
    panel(ax,'B','Control hubs: cortical surfaces')
    ax=fig.add_subplot(grid[1,0])
    top=maps.nlargest(8,'full_hubness').iloc[::-1]
    names=top.roi.str.replace('ctx-lh-','L ',regex=False).str.replace('ctx-rh-','R ',regex=False).str.replace('Left-','L ',regex=False).str.replace('Right-','R ',regex=False)
    names=names.str.replace('superiorfrontal','superior frontal',regex=False).str.replace('superiorparietal','superior parietal',regex=False)
    ax.barh(names,top.full_hubness,color=TEAL,height=.6)
    ax.set_xlabel('Reference hubness (SD)')
    panel(ax,'C','Highest reference hubness')
    fig.canvas.draw()
    fig.set_layout_engine('none')
    renderer=fig.canvas.get_renderer()
    unit=fig.dpi/72
    horizontal_pad,vertical_pad,text_gap=6*unit,3*unit,2*unit
    sizes=[(title.get_window_extent(renderer),body.get_window_extent(renderer))
           for title,body in card_text]
    heights=[a.height+b.height+text_gap+2*vertical_pad for a,b in sizes]
    card_gap=min(6*unit,(cohort_ax.bbox.height-sum(heights))/2)
    assert card_gap >= 0, 'Cohort boxes exceed panel height'
    left=cohort_ax.bbox.x0+.015*cohort_ax.bbox.width
    top=cohort_ax.bbox.y1
    inverse=cohort_ax.transAxes.inverted()
    for (title,body),(a,b),height in zip(card_text,sizes,heights):
        width=max(a.width,b.width)+2*horizontal_pad
        title.set_position(inverse.transform((left+horizontal_pad,top-vertical_pad)))
        body.set_position(inverse.transform((left+horizontal_pad,top-vertical_pad-a.height-text_gap)))
        (x,y),(right,upper)=inverse.transform([[left,top-height],[left+width,top]])
        cohort_ax.add_patch(FancyBboxPatch((x,y),right-x,upper-y,
            transform=cohort_ax.transAxes,boxstyle='round,pad=0,rounding_size=.025',
            facecolor='#F7F9FC',edgecolor='#c5d1df',linewidth=.8,zorder=0,in_layout=False))
        top-=height+card_gap
    save(fig,'fig2_reference')


def scatter_groups(ax, frame, x, y, legend=False):
    controls=frame.loc[frame.isControl]
    patients=frame.loc[~frame.isControl]
    ax.scatter(controls[x],controls[y],s=10,color=ORANGE,
        marker='o',alpha=.65,linewidths=0,label='Controls')
    ax.scatter(patients[x],patients[y],s=10,color=TEAL,marker='o',alpha=.65,
        linewidths=0,label='TLE')
    if legend:
        ax.legend(frameon=False,fontsize=6)


def hubs(frame, primary, separate_centers):
    """Figure 3: FA concentration and adjusted group estimates."""
    fig, axes=plt.subplots(2,2,figsize=(7.1,4.7),layout='constrained')
    ax=axes[0,0]
    scatter_groups(ax,frame,'expected_concentration','observed_concentration',legend=True)
    bounds=[.17,.46]
    ax.plot(bounds,bounds,'--',color='.6',lw=.8)
    ax.set(xlabel='Spatially matched expectation',ylabel='Observed concentration')
    ax.legend(frameon=False,fontsize=6)
    panel(ax,'A','FA deviations near reference hubs')
    ax=axes[0,1]
    boxes(ax,frame,'excess_hub_concentration')
    ax.axhline(0,color='.6',ls='--',lw=.8)
    ax.set_ylabel('Excess concentration (hubness SD)')
    panel(ax,'B','Observed minus expected concentration')
    ax=axes[1,0]
    scatter_groups(ax,frame,'mean_absolute_fa_z','excess_hub_concentration')
    ax.set(xlabel='Whole-brain mean |FA z|',ylabel='Excess hub concentration')
    panel(ax,'C','Relationship with overall FA deviation')
    ax=axes[1,1]
    p=primary.loc[primary.endpoint.eq('excess_hub_concentration')].iloc[0]
    sites=separate_centers.query("endpoint == 'excess_hub_concentration'")
    forest(ax,[p.to_dict()]+sites.to_dict('records'),['Pooled']+sites.SITE.tolist())
    ax.set_xlabel('Adjusted patient − control difference')
    panel(ax,'D','Adjusted group differences')
    save(fig,'fig3_hub_involvement')


def resilience(frame, curve_table, stages, separate_centers):
    """Figure 4: attacks and adjustment checks."""
    fig,axes=plt.subplots(2,2,figsize=(7.1,4.7),layout='constrained')
    ax=axes[0,0]
    for label,color,style in [('Targeted',TEAL,'-'),('Random',BLUE,'--')]:
        curve=curve_table.loc[curve_table.attack.eq(label)]
        x=curve.percent_isolated.to_numpy()
        ax.fill_between(x,curve.hc3_low.to_numpy(),curve.hc3_high.to_numpy(),color=color,alpha=.13,linewidth=0)
        ax.plot(x,curve.beta.to_numpy(),color=color,lw=1.2,ls=style,label=label)
    ax.axhline(0,color='.6',lw=.8)
    ax.set(xlabel='Nodes isolated (%)',ylabel='Normalized efficiency: TLE − control')
    ax.legend(frameon=False,fontsize=6)
    panel(ax,'A','Group differences during removal')
    ax=axes[0,1]
    boxes(ax,frame,'resilience_gap')
    ax.set_ylabel('Mean normalized efficiency gap')
    panel(ax,'B','Individual vulnerability scores')
    ax=axes[1,0]
    d=stages
    forest(ax,d.to_dict('records'),['Unadjusted','Age, sex, site','+ Density and strength','+ Initial efficiency'])
    ax.set_xlabel('Patient − control difference')
    panel(ax,'C','Staged covariate adjustment')
    ax=axes[1,1]
    d=separate_centers.query("endpoint == 'resilience_gap'")
    forest(ax,d.to_dict('records'),d.SITE.tolist())
    ax.set_xlabel('Within-site adjusted difference')
    panel(ax,'D','Separate-center adjusted differences')
    save(fig,'fig4_resilience')


def outcomes(scores, assoc):
    """Figure 5: score distributions and surgical-outcome associations."""
    fig,axes=plt.subplots(2,2,figsize=(7.1,3.8),gridspec_kw={'height_ratios':[1.15,.8]},layout='constrained')
    scores=scores.copy()
    scores['engel_I']=scores.outcome.eq(0)
    for ax,endpoint,letter,title,ylabel in [
            (axes[0,0],'excess_hub_concentration','A','Hub concentration by outcome','Excess concentration (hubness SD)'),
            (axes[0,1],'resilience_gap','B','Vulnerability by outcome','Mean normalized efficiency gap')]:
        boxes(ax,scores,endpoint,group='engel_I',labels=(f'Engel I (n = {scores.engel_I.sum()})',f'Engel II-IV (n = {(~scores.engel_I).sum()})'))
        ax.set_ylabel(ylabel)
        panel(ax,letter,title)
    for ax,endpoint,letter,title in [(axes[1,0],'excess_hub_concentration','C','Adjusted hub concentration'),(axes[1,1],'resilience_gap','D','Adjusted vulnerability')]:
        d=assoc.loc[assoc.endpoint.eq(endpoint)]
        forest(ax,d.to_dict('records'),[f'Reviewed (n = {int(d.iloc[0].n)})',f'One-year (n = {int(d.iloc[1].n)})'])
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlabel('Adjusted Engel II-IV minus I score')
        panel(ax,letter,title)
    save(fig,'fig5_outcomes')


def set_style():
    plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7,'xtick.labelsize':6.5,
        'ytick.labelsize':6.5,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':INK,
        'text.color':INK,'svg.fonttype':'none','pdf.fonttype':42,'svg.hashsalt':'project3_paper'})
