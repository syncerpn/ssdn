#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "option_list.h"
#include "utils.h"

list *read_module_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if(file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list *options = make_list();
    while((line=fgetl(file)) != 0){
        ++ nu;
        strip(line);
        switch(line[0]){
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
            default:
                if(!read_option(line, options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return options;
}

list *read_data_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if(file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list *options = make_list();
    while((line=fgetl(file)) != 0){
        ++ nu;
        strip(line);
        switch(line[0]){
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
            default:
                if(!read_option(line, options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return options;
}

metadata get_metadata(char *file)
{
    metadata m = {0};
    list *options = read_data_cfg(file);

    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", 0);
    if(!name_list) {
        fprintf(stderr, "No names or labels found\n");
    } else {
        m.names = get_labels(name_list);
    }
    m.classes = option_find_int(options, "classes", 2);
    free_list(options);
    return m;
}

int read_option(char *s, list *options)
{
    size_t i;
    size_t len = strlen(s);
    char *val = 0;
    for(i = 0; i < len; ++i){
        if(s[i] == '='){
            s[i] = '\0';
            val = s+i+1;
            break;
        }
    }
    if(i == len-1) return 0;
    char *key = s;
    option_insert(options, key, val);
    return 1;
}

void option_insert(list *l, char *key, char *val)
{
    kvp *p = malloc(sizeof(kvp));
    p->key = key;
    p->val = val;
    p->used = 0;
    list_insert(l, p);
}

void option_unused(list *l)
{
    node *n = l->front;
    while(n){
        kvp *p = (kvp *)n->val;
        if(!p->used){
            fprintf(stderr, "Unused field: '%s = %s'\n", p->key, p->val);
        }
        n = n->next;
    }
}

char *option_find(list *l, char *key)
{
    node *n = l->front;
    while(n){
        kvp *p = (kvp *)n->val;
        if(strcmp(p->key, key) == 0 && p->used != 1){
            p->used = 1;
            return p->val;
        }
        n = n->next;
    }
    return 0;
}
char *option_find_str(list *l, char *key, char *def)
{
    char *v = option_find(l, key);
    if(v) return v;
    if(def) fprintf(stderr, "%s: Using default '%s'\n", key, def);
    return def;
}

char *option_find_str_quiet(list *l, char *key, char *def)
{
    char *v = option_find(l, key);
    if(v) return v;
    return def;
}

void option_find_str_series(list *l, char *key, int* num, char*** series)
{
    char *v = option_find(l, key);
    if (v) {
        int len = strlen(v);
        int n = 1;
        int i, j, k, m;
        for(i = 0; i < len; ++i){
            if (v[i] == ',') ++n;
        }
        
        if (*series == 0) {
            *series = calloc(n, sizeof(char*));
        }

        k = 0;

        for(j = 0; j < len; ++j){
            for (i = j; i < len; ++i) {
                if (v[i] == ',') break;
            }
            
            int sub_bgn = j;
            int sub_end = i;
            for (m = sub_bgn; m < sub_end; ++m) {
                if (v[m] != ' ') break;
            }
            sub_bgn = m;

            for (m = sub_end - 1; m >= sub_bgn; m--) {
                if (v[m] != ' ') break;
            }
            sub_end = m;

            int sublen = sub_end - sub_bgn + 1;

            (*series)[k] = calloc(sublen + 1, sizeof(char));
            memcpy((*series)[k], v+sub_bgn, sublen);
            (*(*series + k))[sublen] = '\0';

            j = i;
            ++k;
        }
        if (num) *num = n;
        return;
    }
    if (num) *num = 0;
    return;
}

int option_find_int(list *l, char *key, int def)
{
    char *v = option_find(l, key);
    if(v) return atoi(v);
    fprintf(stderr, "%s: Using default '%d'\n", key, def);
    return def;
}

int option_find_int_quiet(list *l, char *key, int def)
{
    char *v = option_find(l, key);
    if(v) return atoi(v);
    return def;
}

void option_find_int_series(list *l, char *key, int *num, int** series)
{
    char *v = option_find(l, key);
    if (v) {
        int len = strlen(v);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (v[i] == ',') ++n;
        }
        
        if (*series == 0) {
            *series = calloc(n, sizeof(int));
        }

        for(i = 0; i < n; ++i){
            int val = atoi(v);
            *(*series + i) = val;
            v = strchr(v, ',')+1;
        }
        if (num) *num = n;
        return;
    }
    if (num) *num = 0;
    return;
}

float option_find_float(list *l, char *key, float def)
{
    char *v = option_find(l, key);
    if(v) return atof(v);
    fprintf(stderr, "%s: Using default '%lf'\n", key, def);
    return def;
}

float option_find_float_quiet(list *l, char *key, float def)
{
    char *v = option_find(l, key);
    if(v) return atof(v);
    return def;
}

void option_find_float_series(list *l, char *key, int* num, float** series)
{
    char *v = option_find(l, key);
    if (v) {
        int len = strlen(v);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (v[i] == ',') ++n;
        }

        if (*series == 0) {
            *series = calloc(n, sizeof(float));
        }

        for(i = 0; i < n; ++i){
            float val = atof(v);
            *(*series + i) = val;
            v = strchr(v, ',')+1;
        }
        if (num) *num = n;
        return;
    }
    if (num) *num = 0;
    return;
}